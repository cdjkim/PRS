from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from components.component import ComponentE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Encoder(ComponentE, ABC):
    def __init__(self, config):
        super(Encoder, self).__init__(config)

    @abstractmethod
    def forward(self, images):
        pass


class DummyEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = None

    def forward(self, x):
        return x


