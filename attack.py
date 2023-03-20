import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import sys

from MIFPE import MIFPE_untargeted
from utils import Logger


class Attack():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard',
                 device='cuda', log_path=None, decay_step='linear',
                 scale_value=0.9, n_iter=100, t=1.0):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.device = device
        self.logger = Logger(log_path)
        self.decay_step=decay_step
        self.scale_value = scale_value
        self.n_iter = n_iter
        self.t=t

        self.mifpe = MIFPE_untargeted(self.model, n_restarts=1, n_iter=self.n_iter, verbose=False,
                               eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                               device=self.device, decay_step=self.decay_step, t=self.t)

        from MIFPE import MIFPE_targeted
        self.mifpe_targeted = MIFPE_targeted(self.model, n_restarts=1, n_iter=self.n_iter, verbose=False,
                                                 eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                 seed=self.seed, device=self.device, decay_step=self.decay_step,
                                                 t=self.t)

        if version in ['ce', 'cw', 'dlr', 'mifpe', 'ce_t', 'cw_t', 'dlr_t', 'mifpe_t']:
            self.set_version(version)

    def get_logits(self, x):
        x_input = x
        return self.model(x_input)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            first_minus_second_value = torch.zeros(x_orig.shape[0], dtype=torch.float, device=x_orig.device)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x)
                correct_batch = y.eq(output.max(dim=1)[1])
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                maxk = max((10,))
                pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
                first_minus_second_value_batch = (pred_val_out[:, 0] - pred_val_out[:, 1]).detach().to(first_minus_second_value.device)
                first_minus_second_value[start_idx:end_idx] = first_minus_second_value_batch

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))

            x_adv = x_orig.clone().detach()
            startt = time.time()
            for attack in self.attacks_to_run:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_adv[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x)
                    correct_batch = y.eq(output.max(dim=1)[1])
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                    maxk = max((10,))
                    pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
                    first_minus_second_value_batch = (pred_val_out[:, 0] - pred_val_out[:, 1]).detach().to(
                        first_minus_second_value.device)
                    first_minus_second_value[start_idx:end_idx] = first_minus_second_value_batch
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break
                n_batches = int(np.ceil(num_robust / bs))

                before_sorted_robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)

                first_minus_second_value_robust = first_minus_second_value[before_sorted_robust_lin_idcs]
                sorted_first_minus_second_value, indices_first_minus_second_value = torch.sort(first_minus_second_value_robust, dim=0)
                sorted_robust_lin_idcs = before_sorted_robust_lin_idcs[indices_first_minus_second_value]
                robust_lin_idcs = sorted_robust_lin_idcs

                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)

                    if attack == 'ce':
                        self.mifpe.loss = 'ce'
                        self.mifpe.seed = self.get_seed()
                        _, adv_curr = self.mifpe.perturb(x, y)

                    elif attack == 'cw':
                        self.mifpe.loss = 'cw'
                        self.mifpe.seed = self.get_seed()
                        _, adv_curr = self.mifpe.perturb(x, y)

                    elif attack == 'mifpe':
                        self.mifpe.loss = 'ce'
                        self.mifpe.seed = self.get_seed()
                        _, adv_curr = self.mifpe.perturb(x, y)

                    elif attack == 'dlr':
                        self.mifpe.loss = 'dlr'
                        self.mifpe.seed = self.get_seed()
                        _, adv_curr = self.mifpe.perturb(x, y)

                    elif attack == 'ce_t':
                        self.mifpe.loss = 'ce'
                        self.mifpe_targeted.seed = self.get_seed()
                        _, adv_curr = self.mifpe_targeted.perturb(x, y)

                    elif attack == 'cw_t':
                        self.mifpe.loss = 'cw'
                        self.mifpe_targeted.seed = self.get_seed()
                        _, adv_curr = self.mifpe_targeted.perturb(x, y)

                    elif attack == 'mifpe_t':
                        self.mifpe.loss = 'ce'
                        self.mifpe_targeted.seed = self.get_seed()
                        _, adv_curr = self.mifpe_targeted.perturb(x, y)

                    elif attack == 'dlr_t':
                        self.mifpe.loss = 'dlr'
                        self.mifpe_targeted.seed = self.get_seed()
                        _, adv_curr = self.mifpe_targeted.perturb(x, y)
                    else:
                        raise ValueError('Attack not supported')
                    adv_curr_new = torch.clamp(torch.min(torch.max(adv_curr, x - self.epsilon), x + self.epsilon), 0.0,
                                               1.0)
                    output = self.get_logits(adv_curr_new)
                    false_batch = ~y.eq(output.max(dim=1)[1]).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    x_adv[batch_datapoint_idcs] = adv_curr_new.detach().to(x_adv.device)
                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed '.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%} '.format(robust_accuracy))

        return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = x_orig.shape[0] // bs
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(x_orig, y_orig, bs=bs)
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(adv[c], y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log('robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(c.upper(),
                                                                space, acc_indiv, time.time() - startt))

        return adv

    def set_version(self, version='standard'):

        if version == 'ce':
            self.attacks_to_run = ['ce']

        elif version == 'cw':
            self.attacks_to_run = ['cw']

        elif version == 'dlr':
            self.attacks_to_run = ['dlr']

        elif version == 'mifpe':
            self.attacks_to_run = ['mifpe']

        elif version == 'ce_t':
            self.attacks_to_run = ['ce_t']

        elif version == 'cw_t':
            self.attacks_to_run = ['cw_t']

        elif version == 'dlr_t':
            self.attacks_to_run = ['dlr_t']

        elif version == 'mifpe_t':
            self.attacks_to_run = ['mifpe_t']

        self.mifpe.n_restarts = 1
        self.mifpe_targeted.n_restarts = 1
        self.mifpe_targeted.n_target_classes = 9
