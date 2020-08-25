from tensorboardX import SummaryWriter
from .singleton_model import SingletonModel
from .reservoir import reservoir
import torch
import random
import pprint
import colorful
import numpy as np


class SLabReservoir(SingletonModel):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__(config, writer)
        self.rsvr_size = config['reservoir_size']
        self.replay_multiple = config['replay_multiple']
        self.cur_t = self.config['data_schedule'][0]['subsets'][0][1]  # First Task idx.
        # Set-Up reservoir.
        self.rsvr = reservoir[config['reservoir_name']](config)
        self.rsvr_name = config['reservoir_name']
        self.n = 0

        self.per_cat_grad_info = None

    def learn(self, x, y, t, step=None, cat_str_lst=None, split_cats_dict=None, data_obj=None):
        imgs, cats = x.to(self.device), y.to(self.device)

        # Replay reservoir
        if len(self.rsvr) > 0:
            sample_num = imgs.size(0)
            k = min(len(self.rsvr), sample_num * self.replay_multiple)
            # randomly sample k from the reservoir, and merge it.
            replay_dict = self.rsvr.sample(num=k, model=self.component, aux_info=self.per_cat_grad_info)

            if self.rsvr_name == 'random' or self.rsvr_name == 'prs_mlab':
                merged_imgs = torch.cat([imgs, replay_dict['imgs']], dim=0)
                merged_cats = torch.cat([cats, replay_dict['cats']], dim=0)
            elif 'prs' in self.rsvr_name:
                if self.config['batch_sampler'] == 'random' or self.config['batch_sampler'] == "hard_sampling":
                    merged_imgs = torch.cat([imgs, replay_dict['imgs']], dim=0)
                    merged_cats = torch.cat([cats, replay_dict['cats']], dim=0)
                else:
                    merged_imgs = torch.cat([imgs, replay_dict['imgs']], dim=0)
                    merged_cats = torch.cat([cats, replay_dict['cats']], dim=0)

        else:
            merged_imgs, merged_cats = imgs, cats

        nll = self.component.nll(merged_imgs, merged_cats, step=step)
        weight_decay = self.component.weight_decay_loss()
        self.component.zero_grad()

        # update.
        mean_loss = nll.mean() + self.config['weight_decay'] * weight_decay
        mean_loss.backward()
        self.component.clip_grad()
        self.component.optimizer.step()
        self.component.lr_scheduler.step()

        # Update reservoir
        if self.config['reservoir_size'] > 0:
            self.rsvr.update(imgs=imgs, cats=cats)


        if step % self.config['summary_step'] == 0:
            print('Task: {0} [step: {1}]\t'
                  'Loss {2}))'.format(t, step, mean_loss))

            # summary.write(self.writer, step)
            # self.writer.add_histogram('grad', grad, step)
            self.writer.add_scalar(
                'num_params', sum([p.numel() for p in self.parameters()]),
                step)
            self.writer.add_scalar(
                'train_nll', nll.mean(),
                step)

            train_cat_loss = dict()
            for c in merged_cats:
                c = c.item()
                if c not in train_cat_loss:
                    cat_loss_mean = nll[(merged_cats==c).nonzero().flatten()].mean()
                    train_cat_loss[c] = cat_loss_mean

            for cat, loss in train_cat_loss.items():
                self.writer.add_scalar(
                    'train_nll/{}_loss'.format(cat), loss, step
                )

            if self.config['reservoir_size'] == 0:
                return


            weight_norm = 0.
            for p in filter((lambda p: p.requires_grad), self.parameters()):
                weight_norm += torch.norm(p, p=2).item()
            # print('Weight Norm: {}'.format(weight_norm))
            self.writer.add_scalar(
                'train/norm_weight', weight_norm, step
            )
            self.component.train()
