import math
import torch
import torch.nn as nn

class RSLoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        w_eff = self.weight + (self.alpha / math.sqrt(self.r)) * (self.B @ self.A)
        out = x @ w_eff.t()
        if self.bias is not None:
            out = out + self.bias
        return out

def replace_with_rs_lora_linear(module, r=4, alpha=4.0):
    new_module = module
    if isinstance(module, nn.Linear):
        in_f, out_f = module.in_features, module.out_features
        bias = module.bias is not None
        rs_lora = RSLoraLinear(in_f, out_f, r, alpha, bias)
        rs_lora.weight.data.copy_(module.weight.data)
        if bias:
            rs_lora.bias.data.copy_(module.bias.data)
        new_module = rs_lora
    for name, child in module.named_children():
        setattr(module, name, replace_with_rs_lora_linear(child, r, alpha))
    return new_module
