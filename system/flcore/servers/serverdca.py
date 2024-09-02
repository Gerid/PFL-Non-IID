import copy
import os
import random
import time
from flcore.clients.clientdca import clientDCA
from flcore.servers.serverbase import Server
from threading import Thread
import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats

from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from flcore.trainmodel.models import *


class FedDCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDCA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.cluster_inited = False

        # self.load_model()
        self.args = args
        self.args.load_pretrain = False

        self.cluster_models = {}  # Maps clusters to models

        # self.threshold = args.threshold
        # self.branching_factor = args.branching_factor
        self.drift_threshold = 0
        self.clusters = np.zeros(len(self.clients))
        self.cluster_centroids = [copy.deepcopy(self.global_model)]
        self.client_features = {}

    def kde_estimate(self, data):
        bandwidth = (
            1.06 * data.std() * data.size ** (-1 / 5.0)
        )  # Silverman's rule of thumb
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        return kde

    def collect_intermediate_representations(self):
        for client in self.selected_clients:
            data = client.intermediate_output()  # Get the output as numpy array
            self.client_features[client.id] = self.kde_estimate(data)

    def hk_distance(self, dist1, dist2, num_bins=100):
        # Discretize the distributions
        x = np.linspace(
            min(dist1.dataset.min(), dist2.dataset.min()),
            max(dist1.dataset.max(), dist2.dataset.max()),
            num_bins,
        )
        p = dist1(x)
        q = dist2(x)

        # Calculate HK distance
        hellinger_part = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        # Euclidean distance for Kantorovich part
        kantorovich_part = np.sum(p * (x[:, None] - x[None, :]) ** 2 * q)
        return np.sqrt(hellinger_part + kantorovich_part)

    def detect_drift(self, client_id):
        # Simple drift detection logic
        current_features = self.client_features[client_id]
        historical_features = self.client_features.get(client_id, None)
        if (
            historical_features
            and self.hk_distance(current_features, historical_features)
            > self.drift_threshold
        ):
            return True
        return False

    def initialize_clusters(self):
        self.birch = Birch(
            threshold=self.args.threshold, branching_factor=self.args.branching_factor
        )
        self.birch.fit(list(self.client_features.values()))
        self.clusters = self.birch.predict(list(self.client_features.values()))

    def adjust_clusters(self):
        for client_id, features in self.client_features.items():
            self.birch.partial_fit([features])
            self.clusters[client_id] = self.birch.predict([features])[0]
        self.rebalance_clusters()

    def rebalance_clusters(self):
        for node in self.birch.get_leaf_nodes():
            if node.size > self.args.threshold:
                node.split(self.args.branching_factor)

    def update_cluster_centroids(self, cluster_centroids):
        if cluster_centroids is None:
            self.cluster_centroids = {}
        else:
            self.cluster_centroids = cluster_centroids
        for cluster_id, clients in self.clusters.items():
            self.cluster_centroids[cluster_id] = self.aggregate_cluster_models(clients)

    def aggregate_cluster_models(self, cluster_clients):
        weighted_sum = np.zeros_like(self.global_model.parameters())
        total_data_points = 0

        for client in cluster_clients:
            data_points = client.num_data_points()
            total_data_points += data_points
            weighted_sum += data_points * client.model.parameters()

        return weighted_sum / total_data_points

    def update_global_model(self):
        # Aggregate models based on clusters
        cluster_models = {i: [] for i in set(self.clusters)}
        for i, client in enumerate(self.clients):
            cluster_id = self.clusters[i]
            cluster_models[cluster_id].append(client.model)

        for cluster_id, models in cluster_models.items():
            if models:
                self.cluster_centroids[cluster_id] = np.mean(
                    [model.parameters() for model in models], axis=0
                )

        # Update the global model by averaging cluster centroids
        self.global_model = np.mean(list(self.cluster_centroids), axis=0)

    def train(self):
        if self.args.concept_drift:
            drift_args = self.concept_drift()

        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            for client in self.select_clients:
                cluster_id = self.clusters[client.id]
                client.receive_cluster_model(self.cluster_centroids[cluster_id])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                if self.args.concept_drift:
                    client.current_round = i
                    client.drift_args = drift_args[client.id]
                client.train()
                self.client_features[client.id] = client.intermediate_output()

            if not self.cluster_inited:
                self.initialize_clusters()
            else:
                self.adjust_clusters()

            self.update_global_model()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_pretrain_models()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDCA)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
