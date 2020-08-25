from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod
from components.component import ComponentE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .encoder import Encoder


class ResEncoder(Encoder):

    def __init__(self, config, encoded_image_size=14):
        super().__init__(config)
        self.enc_image_size = encoded_image_size

        self.nb_classes = config['nb_classes']
        if config['pretrained']:
            resnet = torchvision.models.resnet101(pretrained=config['pretrained'])  # pretrained ImageNet ResNet-101
        else:
            resnet = torchvision.models.resnet101(pretrained=config['pretrained'], num_classes=self.nb_classes)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, self.nb_classes)
        # Resize image to fixed size to allow input images of variable size
        # self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=config['fine_tune'])

        self.setup_optimizer()


    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        # freeze first 5 module to compare multi-task performance with vac
        # =====================================
        #for p in self.resnet.parameters():
        #    p.requires_grad = False
        #for c in list(self.resnet.children())[5:]:
        #    for p in c.parameters():
        #       p.requires_grad = fine_tune
        # =======================================
        for p in self.resnet.parameters():
            if not fine_tune:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        #for c in list(self.resnet.children())[5:]:
            # for p in c.parameters():
                # p.requires_grad = fine_tune



