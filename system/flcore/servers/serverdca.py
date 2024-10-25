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

from sklearn.metrics import silhouette_score, davies_bouldin_score
from flcore.trainmodel.models import *

class FedDCA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientDCA)

        self.cluster_inited = False
        self.args = args
        self.args.load_pretrain = False

        self.cluster_models = {}  # Maps clusters to models
        self.drift_threshold = 0
        self.clusters = np.zeros(len(self.clients))
        self.cluster_centroids = [copy.deepcopy(self.global_model)]
        self.client_features = {}

    def kde_estimate(self, data):
        bandwidth = 1.06 * data.std() * data.size ** (-1 / 5.0)
        kde = stats.gaussian_kde(data, bw_method=bandwidth)
        return kde

    def collect_proxy_data(self):
        for client in self.selected_clients:
            data = client.intermediate_output()  # Get intermediate representations
            kde = self.kde_estimate(data)
            proxy_data_points = [kde.resample()[0] for _ in range(len(data))]  # Generate proxy points
            self.client_features[client.id] = proxy_data_points

    def vwc_clustering(self):
        # Variational Wasserstein Clustering initialization using client proxy data points
        proxy_points = [point for client_data in self.client_features.values() for point in client_data]
        # Initialize clusters (for simplicity using centroid-based approach in this example)
        cluster_centers = np.random.choice(proxy_points, self.args.num_clusters, replace=False)
        
        # Power Voronoi diagram setup
        # Apply adaptive clustering dynamics (split/merge based on density)
        for i, client_data in self.client_features.items():
            distances = [np.linalg.norm(client_data - center) for center in cluster_centers]
            closest_center = np.argmin(distances)
            self.clusters[i] = closest_center

    def adaptive_clustering(self):
        # Implement density-based adjustments to clusters: split/merge cells in Power Voronoi diagram
        # Calculate density and recency weights, adjust clusters dynamically
        # Placeholder logic (replace with dynamic split/merge in production)
        cluster_weights = {i: len(np.where(self.clusters == i)[0]) for i in set(self.clusters)}
        for cluster_id, weight in cluster_weights.items():
            if weight > self.args.split_threshold:
                # Split cluster
                new_centroid = np.mean([p for idx, p in enumerate(self.client_features) if self.clusters[idx] == cluster_id], axis=0)
                self.cluster_centroids.append(new_centroid)
            elif weight < self.args.merge_threshold:
                # Merge with nearest cluster
                closest_cluster = min(cluster_weights, key=lambda c: np.linalg.norm(self.cluster_centroids[cluster_id] - self.cluster_centroids[c]))
                self.clusters = np.where(self.clusters == cluster_id, closest_cluster, self.clusters)

    def update_global_model(self):
        # Aggregate models based on clusters
        cluster_models = {i: [] for i in set(self.clusters)}
        for i, client in enumerate(self.clients):
            cluster_id = self.clusters[i]
            cluster_models[cluster_id].append(client.model)

        # Average models in each cluster to update global centroids
        for cluster_id, models in cluster_models.items():
            self.cluster_centroids[cluster_id] = np.mean([model.parameters() for model in models], axis=0)
        self.global_model = np.mean(list(self.cluster_centroids), axis=0)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                cluster_id = self.clusters[client.id]
                client.receive_cluster_model(self.cluster_centroids[cluster_id])

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()
                self.client_features[client.id] = client.intermediate_output()

            if not self.cluster_inited:
                self.vwc_clustering()
                self.cluster_inited = True
            else:
                self.adaptive_clustering()

            self.update_global_model()

            self.Budget.append(time.time() - s_t)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_pretrain_models()

