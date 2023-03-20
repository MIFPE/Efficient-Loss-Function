import numpy as np
import time
import torch
import os
import sys
import math
import torch.nn as nn
import torch.nn.functional as F


class MIFPE_untargeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda', decay_step='linear', t=1.0):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.decay_step = decay_step
        self.t = math.pow(10, t)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def cw_loss(self,x,y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)

        mask = torch.unsqueeze(mask, -1)
        return mask


    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / self.t for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == 'Linf':
            t_noise = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t_noise / (
                t_noise.reshape([t_noise.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
            x_adv = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)

        elif self.norm == 'L2':
            t_noise = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t_noise / (
                    (t_noise ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        elif self.loss == 'cw':
            criterion_indiv = self.cw_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv.requires_grad_()

        acc = self.model(x_adv).max(1)[1] == y

        step_size_begin = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0]).to(
            self.device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()

        for i in range(self.n_iter):
            if self.decay_step == 'linear':
                step_size = step_size_begin * (1 - i / self.n_iter)
            elif self.decay_step == 'cos':
                step_size = step_size_begin * math.cos(i / self.n_iter * math.pi * 0.5)
            elif self.decay_step == 'constant':
                step_size = torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0 / 255.0]).to(
                    self.device).detach().reshape([1, 1, 1, 1])

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    x_adv_input = x_adv
                    outputs = self.model(x_adv_input)
                    out_adv = outputs

                    mask_out_adv = self.check_right_index(out_adv, y)
                    mask_out_adv_grad = torch.unsqueeze(torch.unsqueeze(mask_out_adv.clone(), -1), -1)  # #############

                    scale_output = self.get_output_scale(out_adv.clone().detach())

                    if self.loss == 'mifpe':
                        logits_prev = out_adv / scale_output
                    else:
                        logits_prev = out_adv
                    loss_indiv_prev = criterion_indiv(logits_prev, y)
                    loss_prev = loss_indiv_prev.sum()

                grad += torch.autograd.grad(loss_prev, [x_adv])[0].detach()

            grad /= float(self.eot_iter)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + mask_out_adv_grad * step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                    x_adv_1 = torch.clamp(torch.min(
                        torch.max(x_adv + mask_out_adv_grad * ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                  x - self.eps),
                        x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.
            logits_adv = self.model(x_adv)
            pred = logits_adv.detach().max(1)[1] == y
            acc = torch.min(acc, pred)

            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
        x_best_adv[(pred == 1).nonzero().squeeze()] = x_adv[(pred == 1).nonzero().squeeze()] + 0.
        return acc, x_best_adv

    def perturb(self, x_in,  y_in):
        self.seed = 0
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        x_input = x
        acc = self.model(x_input).max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
            acc_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            #
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
            if self.verbose:
                print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s '.format(
                    counter, acc.float().mean(), time.time() - startt))

        return acc, adv

class MIFPE_targeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, device='cuda',
                 n_target_classes=9, decay_step='linear', t=1.0):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        self.loss = loss
        self.decay_step = decay_step
        self.t = math.pow(10, t)

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)
        mask = torch.unsqueeze(mask, -1)
        return mask

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (
                x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)

    def cw_loss_target(self,x,y,y_target):
        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target])

    def ce_targeted(self, x, y, y_target):
        criterion = nn.CrossEntropyLoss(reduce=False, reduction='none')
        return -criterion(x, y_target)

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item /self.t for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == 'Linf':
            t_noise = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t_noise / (
                t_noise.reshape([t_noise.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t_noise = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t_noise / (
                    (t_noise ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()

        x_input = x
        output = self.model(x_input)
        y_target = output.sort(dim=1)[1][:, -self.target_class]

        x_adv.requires_grad_()
        if self.loss == 'ce':
            criterion_indiv = self.ce_targeted
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss_targeted
        elif self.loss == 'cw':
            criterion_indiv = self.cw_loss_targeted
        else:
            raise ValueError('unknowkn loss')

        acc = self.model(x).max(1)[1] == y

        step_size_begin = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor(
            [2.0]).to(self.device).detach().reshape([1, 1, 1, 1])

        x_adv_old = x_adv.clone()

        for i in range(self.n_iter):
            if self.decay_step == 'linear':
                step_size = step_size_begin * (1 - i / self.n_iter)
            elif self.decay_step == 'cos':
                step_size = step_size_begin * math.cos(i / self.n_iter * math.pi * 0.5)
            elif self.decay_step == 'constant':
                step_size = torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0 / 255.0]).to(
                    self.device).detach().reshape([1, 1, 1, 1])

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    x_adv_input = x_adv
                    outputs = self.model(x_adv_input)
                    out_adv = outputs

                    mask_out_adv = self.check_right_index(out_adv, y)
                    mask_out_adv_grad = torch.unsqueeze(torch.unsqueeze(mask_out_adv.clone(), -1), -1)
                    scale_output = self.get_output_scale(out_adv.clone().detach())

                    if self.loss == 'mifpe_t':
                        logits_prev = out_adv / scale_output
                    else:
                        logits_prev = out_adv
                    loss_indiv_prev = criterion_indiv(logits_prev, y, y_target)
                    loss_prev = loss_indiv_prev.sum()
                grad += torch.autograd.grad(loss_prev, [x_adv])[0].detach()

            grad /= float(self.eot_iter)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + mask_out_adv_grad * step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                    x_adv_1 = torch.clamp(torch.min(
                        torch.max(x_adv + mask_out_adv_grad * ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                  x - self.eps),
                        x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / (
                            (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.

            logits_adv = self.model(x_adv)
            pred = logits_adv.detach().max(1)[1] == y
            acc = torch.min(acc, pred)

            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        x_best_adv[(pred == 1).nonzero().squeeze()] = x_adv[(pred == 1).nonzero().squeeze()] + 0.

        return acc, x_best_adv

    def perturb(self, x_in, y_in):
        self.seed = 0
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        x_input = x
        acc = self.model(x_input).max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for target_class in range(2, self.n_target_classes + 2):
            self.target_class = target_class
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                acc_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s '.format(
                            counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))

        return acc, adv
