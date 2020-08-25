import torch
import pickle
import pprint
import colorful
import os

from tensorboardX import SummaryWriter
from .base import Model
from .reservoir import reservoir
from components import E

from utils import AverageMeter, StatMeter
from sklearn.metrics import precision_score, recall_score, f1_score


class MLabReservoir(Model):
    def __init__(self, config, writer: SummaryWriter):
        super().__init__(config, writer)
        self.name = config['model_name']
        self.config = config
        self.device = config['device']
        self.writer = writer
        self.cur_t = self.config['data_schedule'][0]['subsets'][0][1]  # First Task idx.

        self.replay_multiple = config['replay_multiple']

        if config['e'] is not None:
            self.encoder = E[config['e']](config)
        else:
            raise RuntimeError('Component not specified.')

        self.losses = AverageMeter()
        self.accuracy = AverageMeter()

        # Set-Up reservoir.
        self.rsvr = reservoir[config['reservoir_name']](config)
        self.rsvr_name = config['reservoir_name']

        # Loss function
        self.criterion = torch.nn.MultiLabelSoftMarginLoss().to(self.device)

        # Move to GPU, if available
        self.encoder = self.encoder.to(self.device)

    def forward(self, x):
        pass

    def learn(self, x, y, t, step=None, cat_str_lst=None, split_cats_dict=None, data_obj=None):
        """ obtain loss and backprop. update summary as well.
        """
        # reset the losses and accs on new task.
        if t != self.cur_t:
            self.losses.reset()
            self.accuracy.reset()

        imgs, cats = x.to(self.device), y.to(self.device)

        # sample from reservoir to merge into the current mini_batch.
        if len(self.rsvr) > 0:
            k = int(min(len(self.rsvr), imgs.size(0) * self.replay_multiple))
            replay_dict = self.rsvr.sample(online_stream=cats, num=k)
            if self.rsvr_name == 'random' or self.rsvr_name == 'prs_mlab':
                merged_imgs = torch.cat([imgs, replay_dict['imgs']], dim=0)
                merged_cats = torch.cat([cats, replay_dict['cats']], dim=0)
            elif 'prs' in self.rsvr_name:
                if self.config['batch_sampler'] == 'random':
                    merged_imgs = torch.cat([imgs, replay_dict['imgs']], dim=0)
                    merged_cats = torch.cat([cats, replay_dict['cats']], dim=0)
                else:
                    to_merge_imgs = [imgs]
                    to_merge_imgs.extend(replay_dict['imgs'])
                    to_merge_cats = [cats]
                    to_merge_cats.extend(replay_dict['cats'])
                    merged_imgs = torch.cat(to_merge_imgs, dim=0)
                    merged_cats = torch.cat(to_merge_cats, dim=0)

        else:
            merged_imgs, merged_cats = imgs, cats

        # Forward prop.
        predict = self.encoder(merged_imgs)
        targets = merged_cats
        # Calculate loss
        loss = self.criterion(predict, targets)
        loss = loss.mean()

        predict = torch.sigmoid(predict) > 0.5

        total_relevant_slots = targets.sum().data
        relevant_predict = (predict * targets.float()).sum().data

        acc = relevant_predict / total_relevant_slots
        self.losses.update(loss.item())
        self.accuracy.update(acc)

        # Back prop.
        self.encoder.zero_grad()
        loss.backward()

        # Clip gradients
        self.encoder.clip_grad()

        # Update weights
        self.encoder.optimizer.step()

        self.rsvr.update(imgs=imgs, cats=cats)

        # Print status
        if step % self.config['summary_step'] == 0:
            print('Task: {0} [step: {1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(t, step,
                                                                  loss=self.losses,
                                                                  acc=self.accuracy))
            self.writer.add_scalar(
                'train/loss/task%s' % (t),
                loss, step
            )

        self.cur_t = t

