import torch

class KernelTuner:
    def __init__(self, gp_model, lr=0.01, max_steps=50):
        self.gp_model = gp_model
        self.lr = lr
        self.max_steps = max_steps
    def tune(self):
        lengthscale = torch.tensor(self.gp_model.kernel.lengthscale, requires_grad=True)
        variance = torch.tensor(self.gp_model.kernel.variance, requires_grad=True)
        optimizer = torch.optim.Adam([lengthscale, variance], lr=self.lr)
        X = self.gp_model.X
        Y = self.gp_model.Y
        for _ in range(self.max_steps):
            optimizer.zero_grad()
            self.gp_model.kernel.lengthscale = lengthscale
            self.gp_model.kernel.variance = variance
            K = self.gp_model.kernel(X, X) + self.gp_model.noise * torch.eye(X.size(0))
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(Y, L)
            nll = 0.5*(Y.t()@alpha).trace() + torch.log(torch.diagonal(L)).sum() + 0.5*X.size(0)*torch.log(2*torch.pi)
            nll.backward()
            optimizer.step()
        self.gp_model.kernel.lengthscale = lengthscale.detach()
        self.gp_model.kernel.variance = variance.detach()
        self.gp_model.fit(X, Y)
