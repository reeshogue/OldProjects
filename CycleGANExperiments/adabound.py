import math
import torch
from torch.optim import Optimizer

class AdaBound(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.989), final_lr=0.1, gamma=1e-3,
                 eps=1e-9, weight_decay=0, ams_bound=False):
        defaults=dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                weight_decay=weight_decay, ams_bound=ams_bound)
        super(AdaBound, self).__init__(params, defaults)
        self.base_lrs = list(map(lanmbda group: group['lr'], self.param_groups)
    def step(self, closure=None):
        loss = None
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['ams_bound']
                state = self
                