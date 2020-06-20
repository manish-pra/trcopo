import math
import time

import torch
import torch.autograd as autograd
from torch.distributions import Categorical

from trcopo_optim.utils import zero_grad, conjugate_gradient, general_conjugate_gradient, conjugate_gradient_2trpo, conjugate_gradient_trpo
import scipy.sparse.linalg as lin
from scipy.sparse.linalg import LinearOperator


class TRCoPO(object):
    def __init__(self, max_params, min_params, lam=1, threshold=0.01, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.lam = lam
        self.threshold = threshold
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info

        self.old_x = None
        self.old_y = None
        self.old_AinvC = None

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_cgx, self.norm_cgy, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')
    def getlamda(self, grad_vec, kl_vec, params, esp):
        Ainvg, self.iter_num = conjugate_gradient_2trpo(grad_vec = grad_vec, kl_vec=kl_vec,
                                                 params=params, g=-grad_vec,
                                                 x=None, nsteps=10,residual_tol=1e-6,#*grad_x_vec.shape[0],
                                                 device=self.device)

        g = autograd.grad(kl_vec, params, grad_outputs=Ainvg,retain_graph=True)
        g_vec = torch.cat([g.contiguous().view(-1, 1) for g in g])

        sAs = torch.matmul(Ainvg.transpose(0,1),g_vec)

        step_size = torch.sqrt(torch.abs(2*esp/sAs))
        lamda = 1/step_size
        if sAs<0:
            a=1
        else:
            a=0
        return lamda, a


    def step(self, get_log_prob):

        log_probs1, log_probs2, val1_p = get_log_prob()
        objective = torch.exp(log_probs1 + log_probs2 - log_probs1.detach() - log_probs2.detach()) * (val1_p)
        ob = objective.mean()

        kl = torch.mean(-log_probs1 - log_probs2)

        grad_x = autograd.grad(ob, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_x])
        grad_y = autograd.grad(ob, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_y])

        kl_x = autograd.grad(kl, self.max_params, create_graph=True, retain_graph=True)
        kl_x_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_x])
        kl_y = autograd.grad(kl, self.min_params, create_graph=True, retain_graph=True)
        kl_y_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_y])

        lam1,_ = self.getlamda(grad_x_vec, kl_x_vec, self.max_params, self.threshold)
        lam2,_ = self.getlamda(grad_y_vec, kl_y_vec, self.min_params, self.threshold)

        lamda = torch.min(lam1,lam2)
        lamda = lamda/2
        esp = 100
        self.its = 0
        while esp>self.threshold:
            lamda = lamda*2

            def mv(v):
                p = torch.FloatTensor(v).reshape(2*kl_x_vec.size(0), 1)
                c1 = p[0:int(p.size(0) / 2)]
                c2 = p[int(p.size(0) / 2):p.size(0)]
                self.its = self.its + 1
                # print(self.its)
                tot_grad_xy = autograd.grad(grad_y_vec, self.max_params, grad_outputs=c2, retain_graph=True)
                hvp_x_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_xy])  # B1

                tot_grad_yx = autograd.grad(grad_x_vec, self.min_params, grad_outputs=c1, retain_graph=True)
                hvp_y_vec = torch.cat([g.contiguous().view(-1, 1) for g in tot_grad_yx])  # B2

                B = torch.cat([-hvp_x_vec, hvp_y_vec], dim=0)

                a11 = autograd.grad(kl_x_vec, self.max_params, grad_outputs=c1, retain_graph=True)
                a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11])  # A11
                a22 = autograd.grad(kl_y_vec, self.min_params, grad_outputs=c2, retain_graph=True)
                a22_vec = torch.cat([g.contiguous().view(-1, 1) for g in a22])  # A22

                a1 = a11_vec  # + a12_vec
                a2 = a22_vec  # + a21_vec
                A = torch.cat([a1, a2], dim=0)
                Avp_ = lamda * A + B
                return Avp_

            D = LinearOperator((2*kl_x_vec.size(0), 2*kl_x_vec.size(0)), matvec=mv)

            # if self.old_AinvC is not None:
            #     AinvC = lin.lgmres(D, torch.cat([-grad_x_vec,grad_y_vec],dim=0).clone().detach(), maxiter=2, x0=self.old_AinvC, tol=1e-2)
            # else:
            #     AinvC = lin.lgmres(D, torch.cat([-grad_x_vec, grad_y_vec], dim=0).clone().detach(), maxiter=2, tol=1e-2)
            # self.iter_num = 0

            if self.old_AinvC is not None:
                AinvC = lin.gmres(D, torch.cat([-grad_x_vec,grad_y_vec],dim=0).clone().detach(), maxiter=3,restrt=10, x0=self.old_AinvC, tol=1e-2)
            else:
                AinvC = lin.gmres(D, torch.cat([-grad_x_vec, grad_y_vec], dim=0).clone().detach(), maxiter=3,restrt=10, tol=1e-2)
            self.iter_num = 0
            # print('conv',AinvC[1])

            # AinvC_temp, self.iter_num = conjugate_gradient_trpo(grad_x_vec = grad_x_vec, grad_y_vec=grad_y_vec,
            #                                          kl_x_vec=kl_x_vec, kl_y_vec=kl_y_vec, lam=lamda,
            #                                          max_params=self.max_params, min_params=self.min_params, c=torch.cat([-grad_x_vec,grad_y_vec],dim=0),
            #                                          x=None, nsteps=10,#2*grad_x_vec.shape[0],
            #                                          device=self.device, residual_tol=1)

            AinvC = torch.FloatTensor(AinvC[0]).reshape(2*kl_x_vec.size(0), 1)


            c1 = AinvC[0:int(AinvC.size(0)/2)]
            c2 = AinvC[int(AinvC.size(0)/2):AinvC.size(0)]
            a11 = autograd.grad(kl_x_vec, self.max_params, grad_outputs=c1, retain_graph=True)
            a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11])
            a22 = autograd.grad(kl_y_vec, self.min_params, grad_outputs=c2, retain_graph=True)
            a22_vec = torch.cat([g.contiguous().view(-1, 1) for g in a22])

            a1 = a11_vec #+ a12_vec
            a2 = a22_vec #+ a21_vec
            A = torch.cat([a1,a2],dim=0)

            esp = torch.abs(torch.matmul(AinvC.transpose(0,1),A))
            self.old_AinvC = AinvC
            # break
        # print(AinvC, AinvC_temp)


        cg_x = -c1
        cg_y = -c2

        index = 0
        for p in self.max_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')

        new_log_probs1, _, _ = get_log_prob()

        objective = torch.exp(new_log_probs1 - log_probs1.detach()) * (val1_p)
        ob_ed = objective.mean()
        improve1 = ob_ed - ob

        index = 0
        for p in self.min_params:
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        _, new_log_probs2, _ = get_log_prob()

        objective = torch.exp(new_log_probs2 - log_probs2.detach()) * (val1_p)
        ob_ed = objective.mean()
        improve2 = ob_ed - ob
        stat = 5
        if improve1<0 or improve2>0:
            stat = 0
            cg_x = -cg_x
            index=0
            for p in self.max_params:
                p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            cg_y = -cg_y
            index=0
            for p in self.min_params:
                p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
        if improve1<0 and improve2<0:
                stat=1
        if improve1>0 and improve2<0:
                stat=2
        if improve1<0 and improve2>0:
                stat=3
        if improve1>0 and improve2>0:
                stat=4

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.solve_x = False if self.solve_x else True

        return improve1, improve2, lamda, lam1, lam2, esp, stat, self.its

####################################
class TRPO(object):
    def __init__(self, params, lam=1, esp=0.01, bound=0.01, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.model=params
        self.params = list(params.parameters())
        self.lam = lam
        self.esp = esp
        self.bound = bound
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_cgx, self.iter_num, self.norm_kl
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self,get_logprob):

        log_probs, val1_p = get_logprob()

        objective = torch.exp(log_probs-log_probs.detach()) * (-val1_p)
        loss = objective.mean()

        kl = torch.mean(-log_probs)

        grad = autograd.grad(loss, self.params, retain_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1,1) for g in grad])

        kl_p = autograd.grad(kl, self.params, create_graph=True, retain_graph=True)
        kl_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_p])

        esp = self.esp

        Ainvg, self.iter_num = conjugate_gradient_2trpo(grad_vec = grad_vec, kl_vec=kl_vec,
                                                 params=self.params, g=-grad_vec,
                                                 x=None, nsteps=10,#grad_vec.shape[0],
                                                 device=self.device,residual_tol = 1e-4)

        g = autograd.grad(kl_vec, self.params, grad_outputs=Ainvg,retain_graph=True)
        g_vec = torch.cat([g.contiguous().view(-1, 1) for g in g])

        # a11 = autograd.grad(kl_vec, self.params, grad_outputs=Ainvg, retain_graph=True)
        # a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11]) # A11

        sAs = torch.matmul(Ainvg.transpose(0,1),g_vec)

        # def get_kl():
        #     pi_a2_s = self.model(torch.stack(state))
        #     dist_batch2 = Categorical(pi_a2_s)
        #     log_probs2 = dist_batch2.log_prob(action)
        #
        #     kl = torch.mean(-log_probs2)
        #     return kl

        # trpo_step(self.model, loss, get_kl, 0.01, 0)

        step_size = torch.sqrt(torch.abs(2*esp/sAs))
        improvement = 0

        # AinvC are final parameters
        expected_improvement = torch.matmul(Ainvg.transpose(0,1),-grad_vec)*step_size
        cg_x = step_size*Ainvg
        ratio=0
        if sAs>0:
            index = 0
            for p in self.params:
                p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != cg_x.numel():
                raise ValueError('CG size mismatch')

            log_probs_new, val1_p = get_logprob()

            objective = torch.exp(log_probs_new-log_probs.detach()) * (-val1_p)
            loss_ed = objective.mean()
            improvement= loss_ed-loss
            # a=1
            # if sAs<0:
            #     a = -1
            #     if improvement>0:
            #         a=-2

            ratio = -improvement/expected_improvement
            if improvement>0 or ratio<0.1: # change update
                improvement=0
                index = 0
                for p in self.params:
                    cg_x =-cg_x
                    p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                    index += p.numel()

        if self.collect_info:
            self.norm_gx = torch.norm(grad_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_kl = torch.norm(kl_vec, p=2)
        self.solve_x = False if self.solve_x else True

        return step_size, improvement, ratio, sAs

########################################################

class TRCoPO_ORCA(object):
    def __init__(self, max_model, min_model, lam=1, esp=6, bound=0.01, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.max_model = max_model
        self.min_model = min_model
        self.max_params = list(max_model.parameters())
        self.min_params = list(min_model.parameters())
        self.lam = lam
        self.esp = esp
        self.bound = bound
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
                   self.timer, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')
    def getlamda(self, grad_vec, kl_vec, params, esp):
        Ainvg, self.iter_num = conjugate_gradient_2trpo(grad_vec = grad_vec, kl_vec=kl_vec,
                                                 params=params, g=-grad_vec,
                                                 x=None, nsteps=10,#*grad_x_vec.shape[0],
                                                 device=self.device)

        g = autograd.grad(kl_vec, params, grad_outputs=Ainvg,retain_graph=True)
        g_vec = torch.cat([g.contiguous().view(-1, 1) for g in g])

        sAs = torch.matmul(Ainvg.transpose(0,1),g_vec)

        step_size = torch.sqrt(torch.abs(2*esp/sAs))
        if sAs<0:
            a=1
        else:
            a=0
        return step_size, a

    def step(self, advantage, state1, state2, action1,action2):

        val1_p = advantage
        dist_batch1 = self.max_model(state1)
        log_probs1_inid = dist_batch1.log_prob(action1)
        log_probs1 = log_probs1_inid.sum(1)

        dist_batch2 = self.min_model(state2)
        log_probs2_inid = dist_batch2.log_prob(action2)
        log_probs2 = log_probs2_inid.sum(1)

        objective = torch.exp(log_probs1 + log_probs2 - log_probs1.detach() - log_probs2.detach()) * (val1_p)
        ob = objective.mean()

        kl = torch.mean(-log_probs1 - log_probs2)

        grad_x = autograd.grad(ob, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_x])
        grad_y = autograd.grad(ob, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1,1) for g in grad_y])

        kl_x = autograd.grad(kl, self.max_params, create_graph=True, retain_graph=True)
        kl_x_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_x])
        kl_y = autograd.grad(kl, self.min_params, create_graph=True, retain_graph=True)
        kl_y_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_y])

        lam1,_ = self.getlamda(grad_x_vec, kl_x_vec, self.max_params, 0.01)
        lam2,_ = self.getlamda(grad_y_vec, kl_y_vec, self.min_params, 0.01)

        lamda = torch.min(lam1,lam2)
        lamda = lamda/2
        esp = 10
        while esp>self.esp:  #0.0001
            lamda = lamda*2
            AinvC, self.iter_num = conjugate_gradient_trpo(grad_x_vec = grad_x_vec, grad_y_vec=grad_y_vec,
                                                     kl_x_vec=kl_x_vec, kl_y_vec=kl_y_vec, lam=lamda,
                                                     max_params=self.max_params, min_params=self.min_params, c=torch.cat([-grad_x_vec,grad_y_vec],dim=0),
                                                     x=None, nsteps=10,residual_tol=1e-2,#*grad_x_vec.shape[0],
                                                     device=self.device)
            # print(self.iter_num)

            c1 = AinvC[0:int(AinvC.size(0)/2)]
            c2 = AinvC[int(AinvC.size(0)/2):AinvC.size(0)]
            a11 = autograd.grad(kl_x_vec, self.max_params, grad_outputs=c1, retain_graph=True)
            a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11])
            a22 = autograd.grad(kl_y_vec, self.min_params, grad_outputs=c2, retain_graph=True)
            a22_vec = torch.cat([g.contiguous().view(-1, 1) for g in a22])

            a1 = a11_vec #+ a12_vec
            a2 = a22_vec #+ a21_vec
            A = torch.cat([a1,a2],dim=0)

            esp = torch.abs(torch.matmul(AinvC.transpose(0,1),A))


        cg_x = -c1
        cg_y = -c2

        index = 0
        for p in self.max_params:
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise ValueError('CG size mismatch')

        dist_batch1 = self.max_model(state1)
        new_log_probs1_inid = dist_batch1.log_prob(action1)
        new_log_probs1 = new_log_probs1_inid.sum(1)

        objective = torch.exp(new_log_probs1 - log_probs1.detach()) * (val1_p)
        ob_ed = objective.mean()
        improve1 = ob_ed - ob

        index = 0
        for p in self.min_params:
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_y.numel():
            raise ValueError('CG size mismatch')

        dist_batch2 = self.min_model(state2)
        new_log_probs2_inid = dist_batch2.log_prob(action2)
        new_log_probs2 = new_log_probs2_inid.sum(1)
        # new_log_probs1 = log_probs1_inid.sum(1)

        objective = torch.exp(new_log_probs2 - log_probs2.detach()) * (val1_p)
        ob_ed = objective.mean()
        improve2 = ob_ed - ob
        stat = 5
        if improve1<0 or improve2>0:
            stat = 0
            cg_x = -cg_x
            index=0
            for p in self.max_params:
                p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            cg_y = -cg_y
            index=0
            for p in self.min_params:
                p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
        if improve1<0 and improve2<0:
                stat=1
        if improve1>0 and improve2<0:
                stat=2
        if improve1<0 and improve2>0:
                stat=3
        if improve1>0 and improve2>0:
                stat=4

        if self.collect_info:
            self.norm_gx = torch.norm(grad_x_vec, p=2)
            self.norm_gy = torch.norm(grad_y_vec, p=2)
            self.norm_cgx = torch.norm(cg_x, p=2)
            self.norm_cgy = torch.norm(cg_y, p=2)
        self.solve_x = False if self.solve_x else True

        return improve1, improve2, lamda, lam1, lam2, esp, stat

################################################################################
class TRPO_ORCA(object):
    def __init__(self, params, lam=1, esp=0.01, bound=0.01, device=torch.device('cpu'),
                 solve_x=False, collect_info=True):
        self.model=params
        self.params = list(params.parameters())
        self.lam = lam
        self.esp = esp
        self.bound = bound
        self.device = device
        self.solve_x = solve_x
        self.collect_info = collect_info

        self.old_x = None
        self.old_y = None

    def zero_grad(self):
        zero_grad(self.params)

    def getinfo(self):
        if self.collect_info:
            return self.norm_gx, self.norm_gy, self.norm_px, self.norm_py, self.norm_cgx, self.norm_cgy, \
                   self.timer, self.iter_num
        else:
            raise ValueError(
                'No update information stored. Set collect_info=True before call this method')

    def step(self, advantage, state, action):
        val1_p = advantage
        dist_batch = self.model(state)
        log_probs_inid = dist_batch.log_prob(action)
        log_probs = log_probs_inid.sum(1)

        objective = torch.exp(log_probs-log_probs.detach()) * (-val1_p)
        loss = objective.mean()

        kl = torch.mean(-log_probs)

        grad = autograd.grad(loss, self.params, retain_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1,1) for g in grad])

        kl_p = autograd.grad(kl, self.params, create_graph=True, retain_graph=True)
        kl_vec = torch.cat([g.contiguous().view(-1,1) for g in kl_p])

        esp = self.esp

        Ainvg, self.iter_num = conjugate_gradient_2trpo(grad_vec = grad_vec, kl_vec=kl_vec,
                                                 params=self.params, g=-grad_vec,
                                                 x=None, nsteps=10,#*grad_x_vec.shape[0],
                                                 device=self.device)

        g = autograd.grad(kl_vec, self.params, grad_outputs=Ainvg,retain_graph=True)
        g_vec = torch.cat([g.contiguous().view(-1, 1) for g in g])

        # a11 = autograd.grad(kl_vec, self.params, grad_outputs=Ainvg, retain_graph=True)
        # a11_vec = torch.cat([g.contiguous().view(-1, 1) for g in a11]) # A11

        sAs = torch.matmul(Ainvg.transpose(0,1),g_vec)

        # def get_kl():
        #     pi_a2_s = self.model(torch.stack(state))
        #     dist_batch2 = Categorical(pi_a2_s)
        #     log_probs2 = dist_batch2.log_prob(action)
        #
        #     kl = torch.mean(-log_probs2)
        #     return kl

        # trpo_step(self.model, loss, get_kl, 0.01, 0)

        step_size = torch.sqrt(torch.abs(2*esp/sAs))
        improvement = 0

        # AinvC are final parameters
        expected_improvement = torch.matmul(Ainvg.transpose(0,1),-grad_vec)*step_size
        cg_x = step_size*Ainvg
        ratio=0
        if sAs>0:
            index = 0
            for p in self.params:
                p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                index += p.numel()
            if index != cg_x.numel():
                raise ValueError('CG size mismatch')

            dist_batch = self.model(state)
            log_probs_new_inid = dist_batch.log_prob(action)
            log_probs_new = log_probs_new_inid.sum(1)

            objective = torch.exp(log_probs_new-log_probs.detach()) * (-val1_p)
            loss_ed = objective.mean()
            improvement= loss_ed-loss
            # a=1
            # if sAs<0:
            #     a = -1
            #     if improvement>0:
            #         a=-2

            ratio = -improvement/expected_improvement
            if improvement>0 or ratio<0.1: # change update
                improvement=0
                index = 0
                for p in self.params:
                    cg_x =-cg_x
                    p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
                    index += p.numel()

        if self.collect_info:
            self.norm_gx = torch.norm(grad_vec, p=2)
            # self.norm_cgx = torch.norm(cg_x, p=2)
        self.solve_x = False if self.solve_x else True

        return step_size, improvement, ratio

