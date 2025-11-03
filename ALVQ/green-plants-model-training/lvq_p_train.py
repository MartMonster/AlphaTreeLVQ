import torch

class lvq_p_train():
    def __init__(self, net, lr = 0.001):
        self.net = net
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr = lr, weight_decay = 0)

    def loss_fn(self, x, labels):
        log_p = torch.log(self.net.probabilities(x))
        fn = torch.nn.NLLLoss()
        return fn(log_p, labels)
        
    def train(self, x, labels):
        self.optimizer.zero_grad()    
        loss = self.loss_fn(x, labels)
        loss.backward()        
        self.optimizer.step()
        self.net.post_optimizer()
        return loss
    
    def inference(self, x):
        return self.net.inference(x)