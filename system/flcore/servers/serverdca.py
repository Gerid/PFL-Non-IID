import copy
import random
import time
from flcore.clients.clientdca import clientDCA
from flcore.servers.serverbase import Server
from threading import Thread
import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import Birch

class FedDCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDCA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        args.pretrain_round = 50
        args.autoencoder_model_path = "/enc_path"

        # self.load_model()
        self.args = args
        # 初始化代码...
        # 加载自编码器模型
        self.autoencoder = Autoencoder(args.input_size, args.encoding_dim).to(self.device)
        self.autoencoder.load_state_dict(torch.load(args.autoencoder_model_path))
        self.mse_criterion = nn.MSELoss()  # Using Mean Squared Error loss for reconstruction
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=args.autoencoder_lr)

        self.client_clusters = np.zeros(self.num_clients, dtype=int)
        self.cluster_models = {}  # Maps clusters to models

    def collect_intermediate_representations(self):
        # Collect intermediate representations from all clients
        all_intermediate_reps = []
        for client in self.selected_clients:
            all_intermediate_reps.append(client.intermediate_outputs)  # Assuming this attribute exists
        # Calculate the average of intermediate representations
        avg_intermediate_rep = np.mean(all_intermediate_reps, axis=0)
        return torch.tensor(avg_intermediate_rep, dtype=torch.float).to(self.device)

    def autoencoder_train(self, args=None):
        avg_intermediate_rep = self.collect_intermediate_representations()
        self.autoencoder_optimizer.zero_grad()
        _, decoded = self.autoencoder(avg_intermediate_rep.unsqueeze(0))
        loss = self.mse_criterion(decoded, avg_intermediate_rep.unsqueeze(0))
        loss.backward()
        self.autoencoder_optimizer.step()

    def form_clusters(self, client_models):
        features = [self.autoencoder.encode(client.intermediate_output) for client in client_models]
        birch_model = Birch(n_clusters=None)
        birch_model.fit(features)
        cluster_assignments = birch_model.predict(features)

        # Assign or reassign models to clusters
        for cluster_id in np.unique(self.client_clusters):
            if cluster_id not in self.cluster_models:
                self.cluster_models[cluster_id] = self.allocate_new_model()  # Allocate new model for the cluster
        return cluster_assignments


    def pretrain_model(self):
        self.pretrain_rounds=50
        for i in range(self.pretrain_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Pretraining: Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.pretrain()
                self.autoencoder_train(client)

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
            

        # Save the trained autoencoder model
        torch.save(self.autoencoder.state_dict(), self.args.autoencoder_model_path)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDCA)#FML
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def train(self):
        self.pretrain_train()
            
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            
            if round % self.every_recluster_eps == 0:
                self.form_clusters(self.uploaded_models)

            # 聚类内的模型更新聚合
            self.aggregate_within_clusters()

            # 全局模型更新
            self.aggregate_global_model()

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDCA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            cluster_id = self.client_clusters[client.id]
            
            client.set_parameters(self.cluster_models[cluster_id]) # Assign model from client's cluster

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_models = []
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(client.global_model)

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        # use 1/len(self.uploaded_models) as the weight for privacy and fairness
        for client_model in self.uploaded_models:
            self.add_parameters(1/len(self.uploaded_models), client_model)

    def aggregate_within_clusters(self):
        # Assume self.clusters is a list of Cluster objects,
        # and each Cluster object has a list of client models (client_models)
        # and potentially weights associated with each client model.

        for cluster in self.clusters:  # Loop through each cluster
            cluster_model_aggregated = copy.deepcopy(cluster.client_models[0])
            for param in cluster_model_aggregated.parameters():
                param.data.zero_()

            total_weight = 0
            for client_model in cluster.client_models:
                weight = 1  # Or some other weighting criteria
                total_weight += weight
            
                for agg_param, client_param in zip(cluster_model_aggregated.parameters(), client_model.parameters()):
                    agg_param.data += weight * client_param.data

            # Average the aggregated model parameters for the cluster
            for param in cluster_model_aggregated.parameters():
                param.data /= total_weight

            # Optionally, update the global model or each client's model within the cluster with this aggregated model
            # For updating the global model, you might directly set self.global_model = cluster_model_aggregated
            # Or for updating each client model within the cluster, loop through each client and set their model parameters


    # Assuming implementation of encode in Autoencoder is as follows:
    def encode(self, x):
        # Encode the input using the autoencoder's encoder
        return self.autoencoder.encoder(x)

    def should_recluster(last_loss, current_loss, threshold=0.1):
        return abs(current_loss - last_loss) > threshold

# Definition of the Autoencoder class (assuming args contain necessary dimensions)
class Autoencoder(nn.Module):
    # Autoencoder implementation
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        # Encoder and decoder implementation
        self.encoder = nn.Sequential(
            nn.Linear(args.in_features, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.encoding_dim))
        self.decoder = nn.Sequential(
            nn.Linear(args.encoding_dim, args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size, args.in_features),
            nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
