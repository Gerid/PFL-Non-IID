import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from flcore.clients.clientbase import Client

from utils.data_utils import read_client_data
from utils.concept_drift_utils import *


class clientDCA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.alpha = args.alpha
        self.beta = args.beta
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g, gamma=args.learning_rate_decay_gamma
        )
        self.catch_intermediate_output = True
        self.intermediate_output = None
        self.intermediate_outputs = []
        self.drift_interval = 20
        self.drift_args = None

        self.KL = nn.KLDivLoss()

    def receive_cluster_model(self, cluster_model):
        """Initialize the local model with the parameters of the cluster's centroid model."""
        self.model.set_parameters(cluster_model)

    def get_parameters(self):
        parameters = {}
        for name, param in self.model.state_dict().items():
            parameters[name] = (
                param.detach().cpu().numpy()
            )  # 转换为numpy数组，以便于处理
        return parameters

    def hook_fn(self, module, input, output):
        self.intermediate_outputs.append(output)

    def register_hook(self):
        feature_layer = self.model._modules.get("base")
        if feature_layer:
            return feature_layer.register_forward_hook(self.hook_fn)

    def calculate_intermediate_output_average(self):
        # 计算所有捕获的中间输出的平均值
        if self.intermediate_outputs:
            intermediate_output_avg = torch.mean(
                torch.stack(self.intermediate_outputs), 0
            )
            self.intermediate_output = intermediate_output_avg.clone().detach()
            self.intermediate_outputs = []  # 清空列表以便下一轮训练
        else:
            self.intermediate_output = None

    def simulate_concept_drift(self):
        # Example method to change data distribution or switch datasets
        if self.current_round % self.drift_interval == 0:
            self.dataset = self.get_next_drifted_dataset()

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        if self.catch_intermediate_output:
            hook_handle = self.register_hook()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if i == max_local_epochs:
                    break
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # 根据self.drift_args参数和当前轮次，决定是否加入概念漂移和具体形式
                if self.drift_args is not None:
                    y = simulate_complex_label_swaps(
                        y,
                        self.current_round,
                        self.drift_args["swap_schedule"],
                        self.drift_args["random_schedule"],
                    )

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()

        # self.model.cpu()
        # 移除钩子
        hook_handle.remove()
        # 计算中间表征的平均值
        if self.catch_intermediate_output:
            self.calculate_intermediate_output_average()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self, global_model):
        for new_param, old_param in zip(
            global_model.parameters(), self.global_model.parameters()
        ):
            old_param.data = new_param.data.clone()

    # Define the hook function
    def get_intermediate_output(self, module, input, output):
        self.intermediate_outputs.append(output)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                output_g = self.global_model(x)
                loss = self.loss(output, y) * self.alpha + self.KL(
                    F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)
                ) * (1 - self.alpha)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def generate_changepoint(self):
        change_point_str = "rand"
        drift_together = 1
        train_iteration = 10
        stretch_factor = 5
        num_client = 20

        # Randomly generate a single change point for each clientI
        if change_point_str == "rand":
            if drift_together == 1:
                # cp = np.random.random_sample() * train_iteration
                cp = np.random.randint(1, train_iteration // stretch_factor)
                change_point_per_client = [cp for c in range(num_client)]
            else:
                change_point_per_client = [
                    np.random.randint(1, train_iteration // stretch_factor)
                    for c in range(num_client)
                ]

            # matrix of the concept in the training data for each time, client.
            # restricted to concept changes at time step boundary
            change_point = np.zeros((train_iteration // stretch_factor + 1, num_client))
            for c in range(num_client):
                t = change_point_per_client[c]
                change_point[t:, c] = 1
            # np.savetxt("./../../../data/changepoints/rand.cp", change_point, fmt='%u')
        return change_point

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        train_data = CIFAR100DynamicLabel(
            train_data,
            label_shift_func=lambda label: self.shift_data(
                self.current_epoch, self.total_epochs, label
            ),
        )

        return super().load_train_data(batch_size)

    def shift_data(self, epoch, change_point, label_map):
        if epoch >= change_point:
            label = label_map[label]

    def num_data_points(self):
        return len(self.data_loader.dataset)

    def detect_drift(self, new_loss):
        if self.previous_loss is None:
            return False
        drift_coefficient = max((new_loss - self.previous_loss) / self.previous_loss, 0)
        return drift_coefficient > self.drift_threshold


# class

# def __init__(self, root, train=True, transform=None, target_transform=None, label_shift_func=None):
#     self.dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
#     self.target_transform = target_transform
#     self.label_shift_func = label_shift_func

# def __getitem__(self, index):
#     img, label = self.dataset[index]

#     # 应用标签转换
#     if self.label_shift_func is not None:
#         label = self.label_shift_func(label)

#     if self.target_transform is not None:
#         label = self.target_transform(label)

#     return img, label

# def __len__(self):
#     return len(self.dataset)
