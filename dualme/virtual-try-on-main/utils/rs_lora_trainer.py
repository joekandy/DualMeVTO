import torch
import torch.nn.functional as F

class RSLoraTrainer:
    def __init__(self, model, lr=1e-3, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
    def train_step(self, x, y):
        self.opt.zero_grad()
        x, y = x.to(self.device), y.to(self.device)
        out = self.model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        self.opt.step()
        return loss.item()
