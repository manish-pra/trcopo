import math
import time

import torch
import torch.autograd as autograd

from trcopo_optim.utils import zero_grad

class LOLA(object):
    def __init__(self, max_params, min_params, lr=1e-3, weight_decay=0, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info
        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, 0
            # return self.norm_cgx, self.norm_cgy
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, ob, lp1,lp2):
        grad_x = autograd.grad(lp1, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_x])
        grad_y = autograd.grad(lp2, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_y])
        tot_grad_y = autograd.grad(ob.mean(), self.min_params, create_graph=True, retain_graph=True)
        tot_grad_y = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_y])

        tot_grad_xy = autograd.grad(tot_grad_y, self.max_params, grad_outputs=grad_y_vec, retain_graph=True)
        hvp_x_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_xy]) #tot_xy

        tot_grad_x = autograd.grad(ob.mean(), self.max_params, create_graph=True, retain_graph=True)
        tot_grad_x = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_x])

        tot_grad_yx = autograd.grad(tot_grad_x, self.min_params, grad_outputs=grad_x_vec, retain_graph=True)
        hvp_y_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_yx])

        p_x = torch.add(grad_x_vec, - self.lr * hvp_x_vec)
        p_y = torch.add(grad_y_vec, self.lr * hvp_y_vec)
        cg_x = p_x
        cg_y = p_y

        if self.collect_info:
            self.norm_px = torch.norm(p_x, p=2)
            self.norm_py = torch.norm(p_y, p=2)
            self.timer = time.time()

        if self.collect_info:
            self.timer = time.time() - self.timer

        index = 0
        for p in self.max_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(self.lr * cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')
        index = 0
        for p in self.min_params:
            if self.weight_decay != 0:
                p.data.add_(- self.weight_decay * p)
            p.data.add_(- self.lr * cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.solve_x = False if self.solve_x else True