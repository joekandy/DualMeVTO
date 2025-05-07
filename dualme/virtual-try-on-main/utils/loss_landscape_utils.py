# loss_landscape_utils.py

import torch
import matplotlib.pyplot as plt
import copy

def compute_loss_landscape_2d(model, loss_fn, data_loader, device, steps=10, alpha=0.1):
    original_params = [p.clone() for p in model.parameters() if p.requires_grad]
    directions = []
    with torch.no_grad():
        for _ in range(2):
            direction = []
            for p in model.parameters():
                if p.requires_grad:
                    d = torch.randn_like(p)
                    direction.append(d)
                else:
                    direction.append(None)
            directions.append(direction)
    X = torch.linspace(-alpha, alpha, steps)
    Y = torch.linspace(-alpha, alpha, steps)
    Z = torch.zeros((steps, steps))
    for i, xi in enumerate(X):
        for j, yj in enumerate(Y):
            with torch.no_grad():
                idx = 0
                for param in model.parameters():
                    if param.requires_grad:
                        shift = xi*directions[0][idx] + yj*directions[1][idx]
                        param.copy_(original_params[idx] + shift)
                    idx+=1
            total_loss = 0.0
            for batch in data_loader:
                x_in, y_in = batch
                x_in, y_in = x_in.to(device), y_in.to(device)
                preds = model(x_in)
                l = loss_fn(preds, y_in)
                total_loss += l.item()
            Z[i,j] = total_loss
    idx=0
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.copy_(original_params[idx])
                idx+=1
    return X, Y, Z

def plot_loss_landscape_2d(X, Y, Z):
    Xv, Yv = torch.meshgrid(X, Y, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xv.numpy(), Yv.numpy(), Z.numpy(), cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    plt.show()
