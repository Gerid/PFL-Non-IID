import torch
import os
from torch import nn
import copy
import time
import numpy as np
import math
import random


from collections import OrderedDict
from flcore.clients.clienttrans import clientTrans,Cluster, weight_flatten
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import *
from utils.kmeans import kmeans
from threading import Thread


#git clone https://github.com/subhadarship/kmeans_pytorch
#cd kmeans_pytorch
#pip install --editable .
class FedTrans(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientTrans)


        #cluster_number
        self.every_recluster_eps = args.every_recluster_eps
        self.num_cluster = args.num_cluster

        # model embedding layer
        self.emb_dim = args.emb_dim
        self.attn_dim = args.attn_dim

        

        self.emb_layer = nn.Linear(len(nn.utils.parameters_to_vector(self.global_model.head.parameters())),self.emb_dim).to(self.device)

        # attn init
        self.attn_learning_rate = args.attn_learning_rate
        self.attn_init()
        self.alpha = args.alpha
        self.decay_rate = args.decay_rate

        self.pre_train = args.pre_train


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def attn_init(self):
        self.intra_attn_model = Attn_Model(emb_dim=self.emb_dim).to(self.device) # emb client model + grad
        self.param_list = list(self.emb_layer.parameters())
        self.param_list.extend(list(self.intra_attn_model.parameters()))
        self.attn_optimizer = torch.optim.SGD([
                {'params': self.emb_layer.parameters()},
                {'params': self.intra_attn_model.parameters()},
                 ], lr=self.attn_learning_rate, momentum=0.9)
        self.attn_loss = nn.MSELoss().to(self.device)

    def train(self):
        self.cur_iter = 0
        if self.pre_train:
            model_path = os.path.join("models", self.dataset)
            model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
            print("loading prev model {}".format(model_path))
            model = torch.load(model_path).to(self.device)
            head = copy.deepcopy(model.fc)
            model.fc = nn.Identity()
            self.global_model = LocalModel(model, head)
        for i in range(self.global_rounds+1):
            self.cur_iter = i
            gst_time = time.time()
            self.selected_clients = self.select_clients()
            if(i==0):
                self.send_models()
            else:
                self.send_models(init_head=False)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                start_time = time.time()
                self.evaluate()
                end_time = time.time()
                print("evaluate time cost:{}s".format(end_time-start_time))


            start_time = time.time()
            print("local training....")
            psub_res = []
            for client in self.selected_clients:
                client.train()

                #self.use_dp: bool default is False, indicating do not use dp in client 
                client.emb(self.emb_layer)
                psub_res.append(client.psub)
            print("saving psub_res([psub1, psub2,...,]) to psub_res.pth") 
            torch.save(psub_res, "psub_res.pth")

            print("saved psub_res([psub1, psub2,...,]) to psub_res.pth") 

            end_time = time.time()
            print("clients training time cost:{}s".format(end_time-start_time))

            if i != 0:
                start_time = time.time()
                self.attn_optimize()
                end_time = time.time()
                print("attn_optimize time cost:{}s".format(end_time-start_time))
            else:
                start_time = time.time()
                self.form_cluster(reform=True)
                end_time = time.time()
                print("form_cluster time cost:{}s".format(end_time-start_time))
                # log cluster results
                for idx, clus in enumerate(self.clusters):
                    pass

            """
            TODO: consider if it is better that send cluster centroid model to clients and perform local training
            compared to current method, this method is resources costly but seems more resonable.
            """
            
            self.recluster = True
            if self.recluster == True and i%self.every_recluster_eps == 0 and i!=0:
                self.form_cluster(self.recluster)
            
            for cluster in self.active_clusters:
                cluster.avg_update_model()
                cluster.emb(self.emb_layer)
            
            start_time = time.time()
            for cluster in self.active_clusters:
                self.intra_cluster_agg(cluster)
            end_time = time.time()
            print("intra_cluster_agg time cost:{}s".format(end_time-start_time))
            


            self.receive_models()
            self.aggregate_parameters()
            gend_time = time.time()
            print("iter time cost:{}s".format(gend_time-gst_time))

            self.save_attn_model()
            print("saved attn_model")

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        self.save_results()
        self.save_attn_model()
        self.save_global_model()

    def save_attn_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server_attn" + ".pt")
        attn_models = {'emb_layer':self.emb_layer, 'intra_attn_model':self.intra_attn_model}
        print(attn_models)
        torch.save(attn_models, model_path)

    def send_models(self, init_head=True):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model,init_head)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def form_cluster(self, reform=False):
        #compute similarity k-means
        client_emb_list = [client.emb_vec.data.clone().reshape(1, -1) for client in self.clients]
        client_emb_list = torch.cat(client_emb_list, dim=0)

        if reform:
            self.clusters = [Cluster(c_id, self.global_model) for c_id in range(self.num_cluster)]

        for client_id, cluster_id in enumerate(cluster_res[0]):
           self.clusters[cluster_id].clients.append(self.clients[client_id])

        self.active_clusters = []
        for cluster in self.clusters:
            if(len(cluster.clients)>0):
                self.active_clusters.append(cluster)
        
        client_features = self.compute_client_features()

        #cluster_res = kmeans(X=client_emb_list, num_clusters=self.num_cluster, distance='euclidean',iter_limit=200, device=self.device)
        cluster_res = self.dynamic_clustering(client_features)

        
        #cluster.per_layer is the centroid per_model for clients within cluster

    def attn_optimize(self):
        #loss = 0
        #total_train = 0
        for cluster in self.active_clusters:
            if len(cluster.lvs)==0: ## len==0 means cluster only has one client, thus dont need to optimize attn params
                continue
            delta_thetas = [c.sub_head for c in cluster.clients]
            self.attn_optimizer.zero_grad()
            for lv, delta_theta in zip(cluster.lvs, delta_thetas):
                p_d = torch.autograd.grad(lv, self.param_list, list(delta_theta.values()), retain_graph=True)
                for p, g in zip(self.param_list, p_d):
                    if p.grad is None:
                        p.grad = g
                    else:
                        p.grad = p.grad + g
                torch.nn.utils.clip_grad_norm_(self.param_list, 50)
            for param in self.param_list:
                param.data = param.data - self.attn_learning_rate * param.grad
        #for client in self.selected_clients:
            #total_train += client.train_samples
        #for i, client in enumerate(self.selected_clients):
            #grad = client.model.head.grad.clone().detach()
            #client.model.cur_head.backward(grad)
        
    
    """
    def inter_cluster_agg(self):
        cluster_emb_list = [cluster.emb_vec.clone().reshape(1, -1) for cluster in self.active_clusters]
        x = torch.cat(cluster_emb_list, dim=0).squeeze(1)
        weights = self.inter_attn_model(x)
        res = [cluster_emb_list,weights]
        cluster_model_list = [copy.deepcopy(cluster.model.head) for cluster in self.clusters]
        for i in range(weights.size()[0]):
            w = [weights[i][j] for j in range(weights[i].size()[0])]
            head = self.w_add_params(w, cluster_model_list)
            self.active_clusters[i].model.head = head
        return res
    """

    def intra_cluster_agg(self, cluster):
            # 获取每个客户端的表征向量
            client_emb_list = [client.emb_vec.clone().reshape(1, -1) for client in cluster.clients]

            # 将表征向量拼接成一个矩阵
            x = torch.cat(client_emb_list, dim=0).squeeze(1)

            # 初始化聚类的局部模型参数列表
            cluster.lvs = []

            # 如果聚类中只有一个客户端，则直接返回
            if len(cluster.clients) == 1:
                return

            # 对表征向量进行标准化
            X_mean = torch.mean(x, dim=0)
            X_std = torch.std(x, dim=0)
            X_norm = (x - X_mean) / X_std

            # 使用局部注意力模型计算每个客户端的权重
            weights = self.intra_attn_model(X_norm)
            print("weights:{}".format(weights))

            # 获取每个客户端的子头部参数
            sub_head_list = [] 
            for i in range(len(cluster.clients)):
                sub_head_list.append(cluster.clients[i].sub_head)

            # 去除权重张量的冗余维度
            weights = weights.squeeze(0)

            # 将客户端的子头部参数和对应的权重合并，并更新聚类的局部模型参数
            res = [client_emb_list,weights]
            dc = self.alpha * self.decay_rate**(self.cur_iter/self.global_rounds)#update decay
            for i in range(weights.size()[0]):
                w = weights[i]
                c_dict_with_g, c_dict = self.w_add_parameters(w, sub_head_list)
                cluster.clients[i].add_sub(cluster.clients[i].sub_head,decay=dc-1)
                cluster.clients[i].add_sub(c_dict, decay=dc)
                cluster.lvs.append(list(c_dict_with_g.values()))

            return res

    def cluster_update(self):
        for cluster in self.clusters:
            cluster.update_model()
            cluster.emb_vec = self.emb_layer(cluster.model.head)

    def w_add_params(self, m_weights, models_params):
        
        res = copy.deepcopy(self.global_model.head)
        for param in res.parameters():
            param.data.zero_()
            
        for w, model_params in zip(m_weights, models_params):
            for res_param, model_param in zip(res.parameters(), model_params):
                res_param.data += model_param.data.clone() * w
        return res

    def w_add_parameters(self, w, state_dicts):
        sg = OrderedDict()
        sd = OrderedDict()
        assert(len(state_dicts)>0)
        for w_i, sdi in zip(w, state_dicts):
            for key in sdi.keys():
                if key not in sg.keys():
                    sg[key] = w_i * sdi[key]
                    sd[key] = w_i.data * sdi[key]
                else:
                    sg[key] = sg[key] + w_i * sdi[key]
                    sd[key] = sd[key] + w_i.data* sdi[key]
        return sg, sd

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model.base = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.base.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
    
    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.base.parameters(), client_model.parameters()):
            server_param.data = server_param.data + client_param.data.clone() * w

class Attn_Model(nn.Module):
    def __init__(self, emb_dim=128, attn_dim=128, num_heads=8):
        super(Attn_Model, self).__init__()
        self.emb_dim = emb_dim
        self.attn_dim = attn_dim
        self.query = nn.Linear(emb_dim, attn_dim)
        self.key = nn.Linear(emb_dim, attn_dim)
        #self.inter_LN = nn.LayerNorm(attn_dim)

        # 1-layer attention for simple verify

    def forward(self, x, models=None, prev_models=None):
        #x = self.inter_LN(x) 
        q = self.query(x)
        k = self.key(x)

        q_norm = torch.norm(q, dim=-1, keepdim=True)
        k_norm = torch.norm(k, dim=-1, keepdim=True)

        # 计算余弦相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q_norm * k_norm.transpose(-2, -1))



        #scaled coef removed since we want to diff weight matrix entries
        #scores = torch.matmul(q, k.transpose(-2, -1)) / (self.attn_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights

