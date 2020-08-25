import os
import sys
import copy
import logging
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from colorlog import ColoredFormatter

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import operator as op
from functools import reduce
from sklearn.metrics import average_precision_score


def f1_score_overall(targets, predicts, zero_division=0):

    _op = precision_score_overall(targets, predicts, zero_division)
    _or = recall_score_overall(targets, predicts, zero_division)

    return ((2 * _op * _or) / (_op + _or)) if (_op + _or) > 0 else torch.tensor([zero_division], dtype=torch.float32)


def precision_score_overall(targets, predicts, zero_division=0):

    _nc = (targets * predicts).sum().float()
    _np = predicts.sum().float()

    return (_nc / _np) if _np > 0 else torch.tensor([zero_division], dtype=torch.float32)


def recall_score_overall(targets, predicts, zero_division=0):

    _nc = (targets * predicts).sum().float()
    _ng = targets.sum().float()

    return (_nc / _ng) if _ng > 0 else torch.tensor([zero_division], dtype=torch.float32)


def f1_score_per_class(targets, predicts, zero_division=0):

    _cp = precision_score_per_class(targets, predicts, zero_division)
    _cr = recall_score_per_class(targets, predicts, zero_division)

    nu = (2 * _cp * _cr)
    de = (_cp + _cr)

    nu[de == 0] = zero_division
    de[de == 0] = 1.

    return nu / de


def precision_score_per_class(targets, predicts, zero_division=0):

    _nc = (targets * predicts).sum(axis=0).type(torch.float32)
    _np = predicts.sum(axis=0).type(torch.float32)

    _nc[_np == 0] = zero_division
    _np[_np == 0] = 1.

    return _nc / _np


def recall_score_per_class(targets, predicts, zero_division=0):

    _nc = (targets * predicts).sum(axis=0).type(torch.float32)
    _ng = targets.sum(axis=0).type(torch.float32)

    _nc[_ng == 0] = zero_division
    _ng[_ng == 0] = 1.

    return _nc / _ng


def mean_average_precision(targets, predicts):
    """ note that predicts should be probabilities.
    """
    num_cls = targets.size(1)

    aps = torch.zeros(num_cls)
    for i in range(num_cls):
        target = targets[:, i]
        predict = predicts[:, i]

        aps[i] = average_precision_score(target, predict)
    return aps.mean()


def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
     from Aljundi et al, Gradient Based sample selection for online continual learning, Neurips2019
    """
    grads = torch.Tensor(sum(grad_dims))
    grads = grads.cuda() # check for memory issues.
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
     from Aljundi et al, online cont learning with maximally interfered retrieval, Neurips2019
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
     from Aljundi et al, online cont learning with maximally interfered retrieval, Neurips2019
    """
    new_net=copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters,grad_vector,grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data=param.data - lr*param.grad.data
    return new_net

def add_memory_grad(pp, mem_grads, grad_dims):
    """
        This stores the gradient of a new memory and compute the dot product with the previously stored memories.
        pp: parameters

        mem_grads: gradients of previous memories
        grad_dims: list with number of parameters per layers
     from Aljundi et al, Gradient Based sample selection for online continual learning, Neurips2019

    """

    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)

    if mem_grads is None:
        mem_grads = grads.unsqueeze(dim=0)
    else:
        grads = grads.unsqueeze(dim=0)
        mem_grads = torch.cat((mem_grads, grads), dim=0)

    return mem_grads


# handle pytorch tensors etc, by using tensorboardX's method
# try:
    # from tensorboardX.x2num import make_np
# except ImportError:
def make_np(x):
    return np.array(x).copy().astype('float16')

def average_lst(lst):
    return sum(lst) / len(lst)

# n choose r
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
     from Aljundi et al, Gradient Based sample selection for online continual learning, Neurips2019
    """
    grads = torch.Tensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0
    # ignore freezed parameters
    for param in filter((lambda p: p.requires_grad), pp()):
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def add_memory_grad(pp, mem_grads, grad_dims):
    """
        This stores the gradient of a new memory and compute the dot product with the previously stored memories.
        pp: parameters
        mem_grads: gradients of previous memories
        grad_dims: list with number of parameters per layers
     from Aljundi et al, Gradient Based sample selection for online continual learning, Neurips2019

    """

    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)

    if mem_grads is None:
        mem_grads = grads.unsqueeze(dim=0)
    else:
        grads = grads.unsqueeze(dim=0)
        mem_grads = torch.cat((mem_grads, grads), dim=0)

    return mem_grads


def setup_logger():
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger('example')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


class Lambda(nn.Module):
    def __init__(self, f=None):
        super().__init__()
        self.f = f if f is not None else (lambda x: x)

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def duplicate(x, num):
    return x \
        .unsqueeze(1) \
        .expand(x.size(0), num, *x.size()[1:]) \
        .contiguous() \
        .view(x.size(0) * num, *x.size()[1:])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_yaml_format(indices, step_size):
    return {
        'subsets': [['mnist', i] for i in indices],
        'step': step_size,
        'shuffle': True
    }

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def write(self, writer, title, step, info='avg'):
        writer.add_scalar(title+'/_total', getattr(self, info), step)

    def write_to_excel(self, dst, sheet_name, column_name, info='avg'):
        # TODO separate excel writing and csv writing to different function.
        """
        write data at sheet y, column x
        """
        # get dataframe of current data
        df = pd.DataFrame(data=[getattr(self, info)],
                            index=['total_{}'.format(sheet_name).replace(' ', '')], columns=[column_name])

        csv_lst = []
        if os.path.exists(dst):
            dfs_prev = pd.read_excel(dst, index_col=0, sheet_name=None)
            excel_writer = pd.ExcelWriter(dst)

            for sheet_name_tmp in dfs_prev.keys():
                df_prev = dfs_prev[sheet_name_tmp]

                # if there is already a sheet to write, update it
                if sheet_name == sheet_name_tmp:
                    try:
                        df_prev = df_prev.join(df, how='left', sort=False)
                    except ValueError:
                        # overwrite exsiting column.
                        # note that this block is for debug mode.
                        pass
                df_prev.to_excel(excel_writer, sheet_name=sheet_name_tmp)
                csv_lst.append(df_prev)

            # if sheet isn't exist, make it
            if not sheet_name in dfs_prev.keys():
                df.to_excel(excel_writer, sheet_name=sheet_name)
                csv_lst.append(df)
            excel_writer.close()
        else:
            excel_writer = pd.ExcelWriter(dst)
            df.to_excel(excel_writer, sheet_name=sheet_name)
            csv_lst.append(df)
            excel_writer.close()

        csv_frame = csv_lst[0]
        for i in range(1, len(csv_lst)):
            csv_frame = csv_frame.append(csv_lst[i], sort=False)
        csv_frame.to_csv(dst[:dst.index('.xlsx')]+'.csv')



class Group_AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric per groups
    as well as total values
    """

    def __init__(self, ignore_groups=[]):
        self.data = {}
        self.total = AverageMeter()
        self.ignore = [] + ignore_groups

    def reset(self):
        for d in self.data.values():
            d.reset()
        self.total.reset()

    def update(self, groups, vals, ns):
        total_sum = 0
        total_count = 0
        for g, v, n in zip(groups, vals, ns):
            if g in self.ignore:
                continue
            self.update_by_group(g, v, n)
            total_sum += v
            total_count += 1
        self.total.update(total_sum/total_count, total_count)


    def update_by_group(self, group, val, n=1):
        if group not in self.data:
            self.data[group] = AverageMeter()
        self.data[group].update(val, n)


    def write(self, writer, title, step, info='avg'):
        for group in self.data.keys():
            if group in self.ignore:
                continue
            v = getattr(self.data[group], info)
            writer.add_scalar(title+'/%s'%(group), getattr(self.data[group], info), step)
            writer.add_scalar(title[:-5] + '_' + title[-5:] +'/_total', getattr(self.total, info), step)


    def write_to_excel(self, dst, sheet_name, column_name, info='avg'):
        # TODO separate excel writing and csv writing to different function.
        """
        write data at sheet y, column x
        """
        # get dataframe of current data
        cats_sorted = sorted(self.data.keys())
        df = pd.DataFrame(data=[[getattr(self.data[cat], info)] for cat in cats_sorted] + [[getattr(self.total, info)]],
                            index=cats_sorted + ['total_{}'.format(sheet_name).replace(' ', '')], columns=[column_name])

        csv_lst = []
        if os.path.exists(dst):
            dfs_prev = pd.read_excel(dst, index_col=0, sheet_name=None)
            excel_writer = pd.ExcelWriter(dst)

            for sheet_name_tmp in dfs_prev.keys():
                df_prev = dfs_prev[sheet_name_tmp]

                # if there is already a sheet to write, update it
                if sheet_name == sheet_name_tmp:
                    try:
                        df_prev = df_prev.join(df, how='left', sort=False)
                    except ValueError:
                        # overwrite exsiting column.
                        # note that this block is for debug mode.
                        pass
                df_prev.to_excel(excel_writer, sheet_name=sheet_name_tmp)
                csv_lst.append(df_prev)

            # if sheet isn't exist, make it
            if not sheet_name in dfs_prev.keys():
                df.to_excel(excel_writer, sheet_name=sheet_name)
                csv_lst.append(df)
            excel_writer.close()
        else:
            excel_writer = pd.ExcelWriter(dst)
            df.to_excel(excel_writer, sheet_name=sheet_name)
            csv_lst.append(df)
            excel_writer.close()

        csv_frame = csv_lst[0]
        for i in range(1, len(csv_lst)):
            csv_frame = csv_frame.append(csv_lst[i], sort=False)
        csv_frame.to_csv(dst[:dst.index('.xlsx')]+'.csv')


class StatMeter(object):
    """
    Keeps track of samples and return sum, avg, std values.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.list = list()

    def update(self, val):
        self.list.append(val)

    @property
    def sum(self):
        return np.sum(np.asarray(self.list))

    @property
    def avg(self):
        return self.sum / len(self.list)

    @property
    def std(self):
        return np.std(np.asarray(self.list))

    def write(self, writer, title, step, info='avg'):
        writer.add_scalar(title+'/_total', getattr(self, info), step)

class GroupStatMeter(object):
    """
    Keeps track of samples per groups and return sum, avg, std values of each group
    as well as total values
    """

    def __init__(self, ignore_groups=[]):
        self.data = {}
        self.total = StatMeter()
        self.ignore = [] + ignore_groups

    def reset(self):
        for d in self.data.values():
            d.reset()
        self.total.reset()

    def update(self, groups, vals):
        for g, v in zip(groups, vals):
            if g in self.ignore:
                continue
            self.update_by_group(g, v)
            self.total.update(v)


    def update_by_group(self, group, val):
        if group not in self.data:
            self.data[group] = StatMeter()
        self.data[group].update(val)


    def write(self, writer, title, step, info='avg'):
        for group in self.data.keys():
            if group in self.ignore:
                continue
            v = getattr(self.data[group], info)
            writer.add_scalar(title+'/%s'%(group), getattr(self.data[group], info), step)
            writer.add_scalar(title[:-5] + '_' + title[-5:] +'/_total', getattr(self.total, info), step)






def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k, reduction='mean'):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :param reduction: specifies the reduction to apply to the output
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))

    if reduction == 'mean':
        acc = correct.view(-1).float().sum() * (100.0 / batch_size) # 0D tensor
    elif reduction == 'none':
        acc = correct.float().sum(dim=1) * 100.0
    else:
        raise NotImplementedError

    return acc


def summarize_example_wise(packed_sequnce, batch_size):
    """
    summairze packed_sequnce to example wise

    :param packed_sequnce: pakced_sequence to be aligned example-wise
    :param batch_sizes: caption_lengths for restoring padded matrix for calculating example acc
    """
    x = PackedSequence(packed_sequnce, batch_size)
    x, batch_size = pad_packed_sequence(x, batch_first=True)
    x = x.sum(dim=1) / batch_size.to(x.device)
    return x


class RunningStats(object):
    """Computes running mean and standard deviation
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>

    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0., m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def clear(self):
        self.n = 0.

    def push(self, x, s=None, per_dim=False):
        x = make_np(x)
        if s != None:
            s = make_np(s)
        # process input
        if per_dim:
            self.update_params(x, s=s)
        else:
            for el in x.flatten():
                self.update_params(el, s=s)

    def update_params(self, x, s=None):
        self.n += 1
        if self.n == 1:
            self.m = x
            if s is None:
                self.s = 0.
            else:
                self.s = s
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            if s is None:
                self.s += (x - prev_m) * (x - self.m)
            else:
                self.s += (s - self.s) / self.n

    def __add__(self, other):
        if isinstance(other, RunningStats):
            return RunningStats(self.n+other.n, self.n+other.n, self.s+other.s)
        else:
            self.push(other, per_dim=False)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return '<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>'.format(self.mean, self.std, self.n, self.m, self.s)

    def __str__(self):
        return 'mean={: 2.4f}, std={: 2.4f}'.format(self.mean, self.std)
