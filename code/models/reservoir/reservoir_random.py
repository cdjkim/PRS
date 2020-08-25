from .base import rsvrBase
import random
import torch


class rsvrRandom(rsvrBase):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.rsvr_size = config['reservoir_size']
        self.rsvr = {}
        self.rsvr_cursize = 0
        self.n = 0

        self.is_slab = 'slab' in self.config['model_name']

    def update(self, **args):
        """
        args: (imgs:x , cats: categories)
        """
        # ignore invalid reservoir_size
        if not self.rsvr_size > 0:
            return
        nbatch = len(list(args.values())[0])

        for i in range(nbatch):
            if self.n < self.rsvr_size:
                for k in args.keys():
                    if k not in self.rsvr:
                        if type(args[k]) is torch.Tensor:
                            self.rsvr[k] = torch.zeros((self.rsvr_size, *args[k][i].shape),
                                                       dtype=args[k][i].dtype,
                                                       device=args[k][i].device)
                        else:
                            self.rsvr[k] = [None for _ in range(self.rsvr_size)]

                    self.rsvr[k][self.rsvr_cursize] = args[k][i]
                self.rsvr_cursize += 1

            else:
                m = random.randrange(self.n)
                if m < self.rsvr_size:
                    for k in args.keys():
                        self.rsvr[k][m] = args[k][i]
            self.n += 1
        return

    def _onehot_to_slab(self, onehot):
        return (onehot == 1).nonzero()[:, -1]

    def _multihot_to_idxlist(self, multihot):
        idcs = (multihot == 1).nonzero().flatten().tolist()
        return idcs

    def __len__(self):
        return self.rsvr_cursize

    def sample(self, num, model=None, aux_info=None, online_stream=None):
        items = {}
        idx = torch.tensor(random.sample(range(len(self)), num), dtype=torch.long)
        for k in self.rsvr.keys():
            if type(self.rsvr[k]) is torch.Tensor:
                items[k] = self.rsvr[k][idx]
            else:
                items[k] = [self.rsvr[k][i] for i in idx]

        return items

    def __str__(self):
        return 'class: {}\n \
            max size: {}, cur size: {}, len(rsvr): {}, n: {}'.format(type(self).__name__,
                                                                     self.rsvr_size,
                                                                     len(self),
                                                                     len(list(self.rsvr.values())[0]),
                                                                     self.n)

