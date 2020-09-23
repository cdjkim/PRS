from .base import rsvrBase
import random
import torch
import colorful
import math
import numpy as np
from collections import OrderedDict
from copy import deepcopy


class SubStream():
    """
    SubStream idicating samples with a specific label

    main role
    1. save info(e.g. idx in rsvr/ n / proportion)
    2. add(point) item in rsvr.
    3. remove(un-point) item in rsvr.
    """
    def __init__(self, name):
        self.name = name
        self.idxs = list()
        self._probs = list()
        self.proportion = 0.
        self.n = 0.

    def __len__(self):
        return len(self.idxs) # sum(self._probs)

    def remove(self, idx):
        """
        un-point item in rsvr
        :param idx: item's index
        """
        _id = self.idxs.index(idx)
        self.idxs.pop(_id)
        self._probs.pop(_id)

    def add(self, idx, p):
        """
        point item in rsvr
        :param idx: item's index
        """
        self.idxs.append(idx)
        self._probs.append(p)

    def update_stat(self, n):
        """
        update statistics of this substream(class)
        :param n: the number of new examples
        """
        self.n += n


class SubStream_Container():
    """
    Container for substreams to control them easily

    main role
    1. iterate substream
    2. create substream
    3. return substream
    4. update proportions of substreams
    5. calculate distance between target and actual partitions
    """
    def __init__(self, rsvr_total_size):
        self.rsvr_total_size = rsvr_total_size
        self._data = OrderedDict()

    def __getitem__(self, key):
        if key not in self._data:
            self._data[key] = SubStream(name=key)
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def lsum(self):
        """
        summation of l(total number of examples of substream)
        """
        total = 0.
        for st_idx in self.keys():
            total += len(self[st_idx])
        return total

    def keys(self):
        """
        iterate substreams using its name
        :returns: names of substreams
        """
        return (key for key, value in self._data.items() if len(value) > 0)

    def values(self):
        """
        iterate substreams using its value
        :returns: substreams
        """
        return (value for _, value in self._data.items() if len(value) > 0)

    def items(self):
        """
        iterate substreams
        :returns: names and substreams
        """
        return ((key, value) for key, value in self._data.items() if len(value) > 0)

    def get_deltas(self):
        """
        get distance dictionary
        :returns: substream names and its delta
        """
        lsum = self.lsum()
        return {key: len(value) - value.proportion * lsum for key, value in self.items()}

    def get_probs(self):
        """
        get softmax probabilities of substreams with its delta.
        larger delta larger prob!
        :returns: substream names and its probs
        """
        deltas = self.get_deltas()
        probs = torch.FloatTensor([deltas[key] for key in self.keys()])
        probs = torch.softmax(probs, dim=0)

        return {key: probs[i] for i, key in enumerate(self.keys())}

    def get_diff(self):
        """
        get differences with parget
        :returns: differences
        """
        lsum = self.lsum()
        diff_sum = 0.
        for value in self.values():
            diff_sum += abs(len(value) - value.proportion * lsum)
        return diff_sum

    def update_stats(self, keys, cat):
        """
        update statistics of substreams
        :param keys: substream names to update stats.
        """
        for key in keys:
            self[key].update_stat(cat[key])

    def update_proportions(self, rho):
        """
        update proportion of substreams
        :param rho: power of allocation
        """
        total_n_rho = 0.
        for value in self.values():
            total_n_rho += value.n ** rho

        for value in self.values():
            value.proportion = (value.n ** rho) / total_n_rho


class PRS_mlab(rsvrBase):
    """
    Partitioning Reservoir Sampling

    main role
    1. reservoir maintanence
    2. decide which samples should be in and out
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.device = config['device']

        self.rsvr = dict()
        self.rsvr_total_size = config['reservoir_size']
        self.q = config['q_poa']

        self.substreams = SubStream_Container(self.rsvr_total_size)

        self.n = 0
        self.rsvr_cursize = 0

        self.is_slab = False

    def _slab_to_onehot(self, ys):
        """
        covert single label to one-hot vector
        e.g. 4 - > 0 0 0 1 0 0 ...
        :param ys: single labels
        :returns: one-hot vectors
        """
        y_onehot = torch.zeros(ys.size(0), self.config['nb_classes'], dtype=torch.long, device=self.device)
        return y_onehot.scatter(dim=1, index=ys.unsqueeze(1), value=1)

    def _onehot_to_slab(self, onehots):
        """
        convert one-hot vector to single label
        e.g. 0 0 0 1 0 0 ... -> 4
        :param onehots: one-hot vectors
        :returns: single labels
        """
        return (onehots == 1).nonzero()[:, -1]

    def _multihot_to_idxlist(self, multihot):
        """
        convert multi-hot vector to index list(multi label)
        e.g. 0 1 0 1 0 0 ... -> [1, 4]
        :param multihots: multi-hot vectors
        :returns: multi labels
        """
        idcs = (multihot == 1).nonzero().flatten().tolist()
        return idcs

    def _idxlist_to_multihot(self, idcs):
        """
        convert index list to multi-hot vector
        e.g. [1, 4] -> 0 1 0 1 0 0 ...
        :param idcs: index list
        :returns: multi-hot vector
        """
        multihot = torch.zeros(self.config['nb_classes'], dtype=torch.long, device=self.device)
        multihot[idcs] = 1
        return multihot

    def update(self, imgs, cats, **other_datas):
        """
        update reservoir using observed datas.
        this will make label distributions close to target partition
        :param imgs: x
        :param cats: y
        :param other_datas: {dtype1: data2, dtype2: data2}
        """
        if len(cats.shape) == 1:
            self.is_slab = True
            cats = self._slab_to_onehot(cats)

        # merge imgs, cats and other datas for easy control
        other_datas['imgs'] = imgs
        other_datas['cats'] = cats
        datas = other_datas

        nbatch = len(imgs)
        for s_i in range(nbatch):
            sample = {dtype: datas[dtype][s_i] for dtype in datas.keys()} # sample i
            keys = self._multihot_to_idxlist(cats[s_i]) # substream names of sample i

            if self.n < self.rsvr_total_size:
                for dtype in datas.keys():
                    # pre-allocate rsvr memory
                    if dtype not in self.rsvr:
                        if isinstance(datas[dtype], torch.Tensor):
                            self.rsvr[dtype] = torch.zeros((self.rsvr_total_size, *datas[dtype][s_i].shape),
                                                            dtype=datas[dtype][s_i].dtype,
                                                            device=datas[dtype][s_i].device)
                        else:
                            self.rsvr[dtype] = [None for _ in range(self.rsvr_total_size)]

                self.save_sample(self.n, sample)
                self.rsvr_cursize += 1
            else:
                if self.sample_in(keys):
                    # evict old sample and save new ones
                    self.replace_sample(sample)

            # update stats
            self.n += 1
            self.substreams.update_stats(keys, cats[s_i])
            self.partition()

            if self.n % 2000 == 0:
                print(self)

    def sample_in(self, keys):
        """
        determine sample can be in rsvr
        :param keys: substream names of sample
        :returns: True / False
        """
        probs = [0. for _ in keys]
        negn = [0 for _ in keys]

        for i, key in enumerate(keys):
            mi = self.rsvr_total_size * self.substreams[key].proportion
            ni = self.substreams[key].n
            # prob can't be larger than 1
            probs[i] = mi / ni if ni > mi else 1
            negn[i] = -ni

        probs = torch.FloatTensor(probs)
        weights = torch.FloatTensor(negn)
        weights = torch.softmax(weights, dim=0)

        s = torch.sum(probs * weights)

        return random.choices([True, False], [s, 1 - s])[0]

    def save_sample(self, rsvr_idx, sample=None):
        """
        save sample to rsvr. if sample is None, just pointing occurs
        :param rsvr_idx: reservoir idx. sample will be save at this idx.
        :param sample: sample consisting of various datas(e.g. img and cats)
        """
        if sample is not None:
            # actual saving
            for dtype in sample.keys():
                self.rsvr[dtype][rsvr_idx] = deepcopy(sample[dtype])
        else:
            sample = {'cats': self.rsvr['cats'][rsvr_idx]}

        # pointing
        for key in self._multihot_to_idxlist(sample['cats']):
            self.substreams[key].add(rsvr_idx, sample['cats'][key])

    def remove_sample(self, rsvr_idx):
        """
        remove sample from rsvr
        :param rsvr_idx: idx of sample in rsvr.
        """
        # un-pointing
        for key in self._multihot_to_idxlist(self.rsvr['cats'][rsvr_idx]):
            self.substreams[key].remove(rsvr_idx)

    def replace_sample(self, sample):
        """
        replace given sample with old ones in rsvr.
        :param sample: sample to save in rsvr
        """
        rsvr_idx = self.sample_out()
        self.save_sample(rsvr_idx, sample)

    def sample_out(self):
        """
        evict a sample from rsvr
        :returns: removed sample idx of rsvr
        """
        probs = self.substreams.get_probs()
        selected_key = random.choices(list(probs.keys()), weights=list(probs.values()), k=1)[0]
        # y
        y_mask = torch.zeros(self.rsvr_total_size, dtype=torch.bool, device=self.device) # zeros
        y_mask[self.substreams[selected_key].idxs] = True
        # query
        deltas = self.substreams.get_deltas()
        query = [key for key, delta in deltas.items() if delta <= 0]
        query = self._idxlist_to_multihot(query).to(self.device)
        # k
        y_cats = self.rsvr['cats'][y_mask]
        scores = (1 - y_cats) * query
        scores = scores.sum(axis=1)

        idxs_y = self._multihot_to_idxlist(scores == torch.max(scores)) # index within y
        idxs = self._multihot_to_idxlist(y_mask) # index within m(reservoir)
        k = [idxs[idx_y] for idx_y in idxs_y]
        # z
        best = (k[0], self.rsvr_total_size)
        for idx in k:
            #sample = {dtype: self.rsvr[dtype][idx] for dtype in self.rsvr.keys()}
            self.remove_sample(idx)
            diff = self.substreams.get_diff()
            if diff < best[1]:
                best = (idx, diff)
            self.save_sample(idx, sample=None)

        z = best[0]
        self.remove_sample(z)
        return z

    def sample(self, num, **kargs):
        """
        sample replay batch from rsvr
        :param num: num of samples to be sampled
        :returns: samples
        """
        rsvr_idxs = random.sample(range(len(self)), num)
        samples = dict()
        for dtype in self.rsvr.keys():
            if isinstance(self.rsvr[dtype], torch.Tensor):
                samples[dtype] = self.rsvr[dtype][rsvr_idxs]
            else:
                samples[dtype] = [self.rsvr[dtype][i] for i in rsvr_idxs]

        if self.is_slab:
            samples['cats'] = self._onehot_to_slab(samples['cats'])
        return samples

    def partition(self):
        """
        control partitions of the buffer
        """
        self.substreams.update_proportions(self.q)

    def __len__(self):
        return self.rsvr_cursize

    def __str__(self):
        probs = self.substreams.get_probs()
        deltas = self.substreams.get_deltas()
        max_key, max_prob = sorted(probs.items(), key=(lambda x: x[1]))[-1]

        info = colorful.bold_cyan("total substream: {}".format(len(self.substreams))).styled_string
        for key, value in self.substreams.items():

            _info = "\nsubstream {}:\t examples {}\t delta: {:.2f}\t out-prob: {:.2f}".format(
                key, len(value), deltas[key], probs[key])
            if key == max_key:
                _info = colorful.bold_red(_info).styled_string
            else:
                _info = colorful.bold_cyan(_info).styled_string
            info += _info
        return info
