from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class Component(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device'] if 'device' in config else 'cuda'

        self.optimizer = NotImplemented
        self.lr_scheduler = NotImplemented

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = self.config['clip_grad']
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))

    @staticmethod
    def build_optimizer(optim_config, params):
        return getattr(torch.optim, optim_config['type'])(
            params, **optim_config['options'])

    @staticmethod
    def build_lr_scheduler(lr_config, optimizer):
        return getattr(torch.optim.lr_scheduler, lr_config['type'])(
            optimizer, **lr_config['options'])

    def weight_decay_loss(self):
        loss = torch.zeros([], device=self.device)
        for param in self.parameters():
            loss += torch.norm(param) ** 2
        return loss


class ComponentD(Component, ABC):
    def setup_optimizer(self):
        self.optimizer = self.build_optimizer(
            self.config['optimizer_d'], self.parameters())
        self.lr_scheduler = self.build_lr_scheduler(
            self.config['lr_scheduler_d'], self.optimizer)

    class Placeholder(Component, ABC):
        def __init__(self, config):
            super().__init__(config)
            self.p = nn.Parameter(torch.zeros([]), requires_grad=False)
            self.optimizer = self.build_optimizer(
                self.config['optimizer_d'], self.parameters())
            if self.config['lr_scheduler_d']['type'] == 'LambdaLR':
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda=(lambda step: 0 if step == 0 else 1 / step)
                )
            else:
                self.lr_scheduler = self.build_lr_scheduler(
                    self.config['lr_scheduler_d'], self.optimizer)

        def forward(self, x):
            return self.dummy_out.expand(x.size(0), -1)



class ComponentE(Component, ABC):
    def setup_optimizer(self):
        if self.config['optimizer_e']['type'] == 'Adam':
            self.optimizer = self.build_optimizer(
                self.config['optimizer_e'], self.parameters())

        elif self.config['optimizer_e']['type'] == 'SGD':
            self.optimizer = self.build_optimizer(
                self.config['optimizer_e'], self.parameters())
            self.lr_scheduler = self.build_lr_scheduler(
                self.config['lr_scheduler_e'], self.optimizer)
