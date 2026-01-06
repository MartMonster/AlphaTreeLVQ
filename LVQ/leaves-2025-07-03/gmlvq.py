
import torch
import torch.nn as nn
import numpy as np
import math

class gmlvq_net(nn.Module):
    def __init__(self, n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std):
        super().__init__()
        self.protos = nn.Parameter(initial_protos.clone(), requires_grad=True)        
        self.use_matrix_per_proto = use_matrix_per_proto
        self.mu = nn.Parameter(mu.clone(), requires_grad=False)
        stdi = 1.0 / std.clone()
        stdi[std == 0.0] = 0
        self.stdi = nn.Parameter(stdi, requires_grad=False)

        mat = torch.eye(n_feats, requires_grad=False) / np.sqrt(n_feats)
        if use_matrix_per_proto:
            mat = mat.repeat(initial_protos.shape[0], 1, 1)
        self.mat = nn.Parameter(mat, requires_grad=True)
        
        self.proto_labels = torch.tensor(proto_labels)

    def matrix(self, proto_nr):
        if self.use_matrix_per_proto:
            mat = self.mat[proto_nr]
        else:
            mat = self.mat            

        # Ensure positive semi-definite
        return mat @ mat.T 
    
    def proto_distances(self, x):
        x = (x - self.mu) * self.stdi

        distances = []
        for i in range(self.protos.shape[0]):
            mat = self.matrix(i)
            diff = self.protos[i] - x
            result = (diff * (diff @ mat.T)).sum(axis = 1)
            distances.append(result)
        distances = torch.stack(distances, axis = 1)            
        return distances

    def forward(self, x):
        with torch.no_grad():
            distances = self.proto_distances(x)
            return self.proto_labels[distances.argmin(1)]

class gmlvq():
    def __init__(self, LR=1e-4):
        self.LR = LR

    def optimizer(self):
        return torch.optim.AdamW([
        {"params": self.net.protos, "lr": self.LR},
        {"params": self.net.mat, "lr": self.LR * 0.1},
    ], weight_decay=0.0)

    def initialize(self, n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std):
        self.net = gmlvq_net(n_feats, initial_protos, proto_labels, use_matrix_per_proto, mu, std)
        self.optimizer = self.optimizer()

    def save(self, file):
        torch.save(self.net, file)

    def load(self, file):
        self.net = torch.load(file, weights_only=False)
        self.optimizer = self.optimizer()

    def loss_fn(self, x, labels):
        # distances between features and prototypes
        distances = self.net.proto_distances(x)
        # correct classes
        correct = self.net.proto_labels.unsqueeze(0).eq(labels.unsqueeze(1))
        # minimum distance to a prototype with the same label
        min_distance_same = distances.maximum((~correct) * np.finfo(np.float32).max).min(1).values
        # minimum distance to a prototype with a different label
        min_distance_other = distances.maximum((correct) * np.finfo(np.float32).max).min(1).values
        
        eps = 1e-8
        loss = ((min_distance_same - min_distance_other) /
            (min_distance_same + min_distance_other + eps)).mean()

        return loss
    
    def train(self, x, labels):
        self.optimizer.zero_grad()    
        loss = self.loss_fn(x, labels)
        loss.backward()        
        self.optimizer.step()
        
        with torch.no_grad():
            if self.net.use_matrix_per_proto:
                for i in range(self.net.protos.shape[0]):
                    covariance_matrix = self.net.mat[i] @ self.net.mat[i].T
                    trace = torch.trace(covariance_matrix)
                    if trace > 0:
                        self.net.mat[i] /= torch.sqrt(trace)
            else:
                covariance_matrix = self.net.mat @ self.net.mat.T
                trace = torch.trace(covariance_matrix)
                if trace > 0:
                    self.net.mat /= torch.sqrt(trace)
            
        return loss
    
    def inference(self, x):
        return self.net(x)
    
    def freeze_metric(self):
        self.net.mat.requires_grad_(False)

    def unfreeze_metric(self):
        self.net.mat.requires_grad_(True)
