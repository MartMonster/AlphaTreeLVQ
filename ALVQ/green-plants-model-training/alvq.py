import torch
import torch.nn as nn
import numpy as np
import math

class alvq_net(nn.Module):
    def __init__(self, n_feats, initial_protos, use_matrix_per_proto, mu, std):
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
        
    def matrix(self, proto_nr):
        if self.use_matrix_per_proto:
            mat = self.mat[proto_nr]
        else:
            mat = self.mat            

        # Ensure positive semi-definite
        return mat.T @ mat       

    def length_in_space(self, mat, x):
        n_feats = x.shape[-1]
        mapped_x = (mat @ x.reshape(-1, n_feats, 1)).reshape(1, -1, n_feats)
        x = x.reshape(1, -1, n_feats) * mapped_x
        return torch.sqrt(torch.clamp(x.sum(2), min=1e-12)).squeeze()

    def probabilities(self, x):
        x = self.angular_dissimilarities(x)
        if torch.isnan(x).any() or torch.isinf(x).any() or (x == 0.0).any():
            print("angular dissimilarities contain NaN, Inf or negative values")
            print(x)
            raise RuntimeError("angular dissimilarities contain NaN, Inf or negative values")
        beta = 1
        x = (torch.exp(-beta * (x - 1)) - 1) / (np.exp(beta * 2) - 1)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("probabilities contain NaN or Inf before normalization")
            print(x)
            raise RuntimeError("probabilities contain NaN or Inf before normalization")
        x = x / torch.clamp(x.sum(1, keepdim=True), min=1e-12)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("probabilities contain NaN or Inf after normalization")
            print(x)
            raise RuntimeError("probabilities contain NaN or Inf after normalization")
        x = torch.clamp(x, min=0.001, max=0.999)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("probabilities contain NaN or Inf")
            print(x)
            raise RuntimeError("probabilities contain NaN or Inf")
        return x         
    
    def angular_dissimilarities(self, x):
        n_feats = x.shape[-1]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("input contains NaN or Inf")
            print(x)
            raise RuntimeError("input contains NaN or Inf")
        if (x.dim() > 2):
            x = x.reshape(-1, n_feats)        

        x = (x - self.mu) * self.stdi

        similarities = []
        for i in range(self.protos.shape[0]):
            mat = self.matrix(i)
            x_lens = self.length_in_space(mat, x)
            proto_len = self.length_in_space(mat, self.protos[i])
            lens = x_lens * proto_len
            mapped_proto = (mat @ self.protos[i].reshape(n_feats, 1)).reshape(1, n_feats)
            z = x * mapped_proto
            sim = z.sum(1) / (lens + 1e-7)
            similarities.append(sim)
            
        similarities = torch.stack(similarities, axis = 1)                    
        
        return similarities        

    def post_optimizer(self):
        # normalize matrices
        with torch.no_grad():
            if self.use_matrix_per_proto:
                for i in range(self.protos.shape[0]):
                    self.mat[i] /= torch.sqrt(self.matrix(i).trace())
            else:
                self.mat /= torch.sqrt(self.matrix(0).trace())

    def inference(self, x):
        with torch.no_grad():
            return self.angular_dissimilarities(x).argmin(1)