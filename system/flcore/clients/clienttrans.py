import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
from collections import OrderedDict
import numpy as np
import time
import copy


class clientTrans(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.cluster_id = None
        self.emb_vec = None
        self.prev_head = []
        self.psub = None
        self.loss = nn.CrossEntropyLoss()

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        self.phead = copy.deepcopy(self.model.head)
        #for param in self.model.base.parameters():
            #param.requires_grad = False
        #for param in self.model.head.parameters():
            #param.requires_grad = True
        
        initial_params = {name: param.clone() for name, param in self.model.head.named_parameters()}

        for step in range(max_local_steps):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)

                loss.backward()
                self.optimizer.step()
        self.get_subhead()
                
         
        del trainloader


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def get_subhead(self):
        self.psub = []
        self.sub_head = {}
        for name, param1 in self.model.head.named_parameters():
            param2 = self.phead.state_dict()[name]
            diff = param1.data.clone() - param2.data.clone()
            self.sub_head[name] = diff
            self.psub.append(diff.view(-1))
        self.psub = torch.cat(self.psub, dim=0)


    def emb(self, emb_layer:nn.modules, use_dp=False):
        params = []
        for p in self.phead.parameters():
            params.append(p.flatten())
        params = torch.cat(params)
        #emb_m = emb_layer(params) 
        emb_g = emb_layer(self.psub)

        self.emb_vec = emb_g 
        #self.emb_vec = torch.cat([emb_m, emb_g])
    
    def DP_emb(self, emb_layer:nn.modules):
        params = []
        for p in self.phead.parameters():
            params.append(p.flatten())
        params = torch.cat(params)
        emb_m = emb_layer(params) 
        emb_g = emb_layer(self.psub)
        self.emb_vec = torch.cat([emb_m, emb_g])
    

    def add_sub(self, sub, decay=1):
        for n, p in self.model.head.named_parameters():
            p = p.detach() + sub[n].detach() * decay
            
    

    def set_parameters(self, model,init_head=True):
        if init_head:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()
        else:    
            for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()

    #def set_parameters(self, model, coef_self):
        #for new_param, old_param in zip(model.parameters(), self.client_u.parameters()):
            #old_param.data = (new_param.data + coef_self * old_param.data).clone()


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)

    return params

class Cluster():
    def __init__(self, cluster_id, model):
        self.cluster_id = cluster_id
        self.clients = []
        self.model = copy.deepcopy(model)
        self.emb_vec = None
        self.psub = None
        self.active = False
        self.selected_clients = []
    
    def avg_update_model(self):
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_psubs = []
        tot_samples = 0
        for client in self.clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + client.send_time_cost['num_rounds']
            #if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            self.uploaded_psubs.append(client.psub)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


        if(len(self.uploaded_models) > 0):
            self.active =True
        self.model = copy.deepcopy(self.uploaded_models[0])
        for param in self.model.parameters():
            param.data.zero_()
        
        self.psub = torch.zeros_like(self.uploaded_psubs[0])
            
        for w, client_model,client_psub in zip(self.uploaded_weights, self.uploaded_models, self.uploaded_psubs):
            self.psub = self.psub + w * client_psub
            for cluster_param, client_param in zip(self.model.parameters(), client_model.parameters()):
                cluster_param.data = cluster_param.data + client_param.data.clone() * w


    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])


    def emb(self, emb_layer:nn.modules):
        params = []
        for p in self.model.head.parameters():
            params.append(p.flatten())
        params = torch.cat(params)
        #emb_m = emb_layer(params)
        emb_g = emb_layer(self.psub)
        self.emb_vec = emb_g
        #self.emb_vec = torch.cat([emb_m, emb_g])