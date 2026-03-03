import torch

class lvq_p_train():
    def __init__(self, net, lr = 0.001):
        self.net = net
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = lr, weight_decay = 0)

    def loss_fn(self, x, labels):
        log_p = torch.log(self.net.probabilities(x))
        if torch.isnan(log_p).any() or torch.isinf(log_p).any():
            print("log probabilities contain NaN or Inf")
            print(log_p)
            raise RuntimeError("log probabilities contain NaN or Inf")
        fn = torch.nn.NLLLoss()
        return fn(log_p, labels)
        
    def train(self, x, labels):
        self.optimizer.zero_grad()    
        if torch.isnan(self.net.mat).any():
            print("matrix contains NaN 1")
            print(self.net.mat)
            raise RuntimeError("matrix contains NaN")
        loss = self.loss_fn(x, labels)
        if torch.isnan(self.net.mat).any():
            print("matrix contains NaN 2")
            print(self.net.mat)
            raise RuntimeError("matrix contains NaN")
        loss.backward()        
        self.optimizer.step()
        if torch.isnan(self.net.mat).any():
            print("matrix contains NaN 3")
            print(self.net.mat)
            raise RuntimeError("matrix contains NaN")
        self.net.post_optimizer()
        if torch.isnan(self.net.mat).any():
            print("matrix contains NaN 4")
            print(self.net.mat)
            raise RuntimeError("matrix contains NaN")
        return loss
    
    def inference(self, x):
        return self.net.inference(x)