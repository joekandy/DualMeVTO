import torch
import torch.nn as nn
from rs_lora_modules import replace_with_rs_lora_linear
from rs_lora_trainer import RSLoraTrainer

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def main():
    model = SimpleClassifier()
    model = replace_with_rs_lora_linear(model, r=16, alpha=16.0)
    trainer = RSLoraTrainer(model, lr=1e-3)
    for step in range(100):
        x = torch.randn(64, 16)
        y = torch.randint(0, 10, (64,))
        loss_val = trainer.train_step(x, y)
    print("Final loss:", loss_val)

if __name__ == "__main__":
    main()
