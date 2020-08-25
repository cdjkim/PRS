from abc import ABC, abstractmethod
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .component import ComponentD
from utils import Lambda


class Classifier(ComponentD, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.ce_loss = nn.NLLLoss(reduction='none')
        self.config = config

    @abstractmethod
    def forward(self, x):
        """Output log P(y|x)"""
        pass

    def nll(self, x, y, step=None):
        x, y = x.to(self.device), y.to(self.device)
        log_softmax = self.forward(x)
        loss_pred = self.ce_loss(log_softmax, y)
        return loss_pred

    class Placeholder(ComponentD.Placeholder):
        """Dummy classifier assigning probability 1 for any data"""

        def __init__(self, config):
            super().__init__(config)
            self.dummy_out = torch.ones([1, config['y_c']], device=self.device)
            self.to(self.device)

        def forward(self, x):
            return self.dummy_out.expand(x.size(0), -1)

        def nll(self, x, y, step=None, temp=None):
            return torch.zeros([x.size(0)], device=self.device)


class MlpClassifier(Classifier):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config['x_c'] * config['x_h'] * config['x_w'], config['h1_dim'])
        self.fc2 = nn.Linear(config['h1_dim'], config['h2_dim'])
        self.fc3 = nn.Linear(config['h2_dim'], config['y_c'])

        self.to(self.device)
        self.setup_optimizer()

    def forward(self, x):
        x = x.to(self.device).view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out,dim=1)
        # return self.net(x)

        return out

    def intermediate_forward(self, x, layer_index):
        x = x.view(x.size(0), -1)
        if layer_index == 0:
            out = x

        elif layer_index == 1:
            out = F.relu(self.fc1(x))

        elif layer_index == 2:
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))

        return out


class MlpClassifier4(Classifier):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config['x_c'] * config['x_h'] * config['x_w'], config['h1_dim'])
        self.fc2 = nn.Linear(config['h1_dim'], config['h2_dim'])
        self.fc3 = nn.Linear(config['h2_dim'], config['h3_dim'])
        self.fc4 = nn.Linear(config['h3_dim'], config['y_c'])

        self.to(self.device)
        self.setup_optimizer()

    def forward(self, x):
        x = x.to(self.device).view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.log_softmax(out,dim=1)
        # return self.net(x)

        return out

    def intermediate_forward(self, x, layer_index):
        x = x.view(x.size(0), -1)
        if layer_index == 0:
            out = x

        elif layer_index == 1:
            out = F.relu(self.fc1(x))

        elif layer_index == 2:
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, upsample=None,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        transpose = upsample is not None and stride != 1
        self.conv1 = (
            conv4x4t(inplanes, planes, stride) if transpose else
            conv3x3(inplanes, planes, stride)
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        elif self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv4x4t(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """4x4 transposed convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size=4, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )

class ResNetClassifier(Classifier):
    block = BasicBlock
    num_blocks = [2, 2, 2, 2]

    def __init__(self, config):
        super(ResNetClassifier, self).__init__(config)
        if 'num_blocks' in config:
            num_blocks = config['num_blocks']
        else:
            num_blocks = self.num_blocks
        if 'norm_layer' in config:
            self.norm_layer = getattr(nn, config['norm_layer'])
        else:
            self.norm_layer = nn.BatchNorm2d
        num_classes = config['nb_classes']
        self.nf = nf = config['h1_dim']
        self.conv1 = nn.Conv2d(
            3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = self.norm_layer(nf * 1)
        self.layer1 = self._make_layer(nf * 1, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(nf * 1, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(nf * 2, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(nf * 4, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8, num_classes)
        self.to(self.device)
        self.setup_optimizer()

    def _make_layer(self, nf_in, nf_out, num_blocks, stride):
        norm_layer = self.norm_layer
        block = self.block
        downsample = None
        if stride != 1 or nf_in != nf_out:
            downsample = nn.Sequential(
                conv1x1(nf_in, nf_out, stride),
                norm_layer(nf_out),
            )
        layers = [block(
            nf_in, nf_out, stride, downsample=downsample, norm_layer=norm_layer
        )]
        for _ in range(1, num_blocks):
            layers.append(block(nf_out, nf_out, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return nn.functional.log_softmax(out, dim=1)


class CnnClassifier(Classifier):
    def __init__(self, config):
        super().__init__(config)
        feature_volume = (
                (config['x_h'] // 4) *
                (config['x_w'] // 4) *
                config['h2_dim'])

        self.net = nn.Sequential(
            nn.Conv2d(config['x_c'], config['h1_dim'],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config['h1_dim'], config['h2_dim'],
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Lambda(lambda x: x.view(x.size(0), -1)),
            nn.Linear(feature_volume, config['fc_dim'], bias=False),
            nn.ReLU(),
            nn.Linear(config['fc_dim'], config['y_c']),
            nn.LogSoftmax(dim=1)
        )
        self.to(self.device)
        self.setup_optimizer()

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x)


    def intermediate_forward(self, x, layer_index):
        if layer_index == 0:
            interm_net = nn.Sequential(*list(self.net.children())[:2])
            out = interm_net(x)

        elif layer_index == 1:
            interm_net = nn.Sequential(*list(self.net.children())[:4])
            out = interm_net(x)

        elif layer_index == 2:
            interm_net = nn.Sequential(*list(self.net.children())[:-2])
            out = interm_net(x)

        return out


    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        interm_net = nn.Sequential(*list(self.net.children())[:2])
        out = interm_net(x)
        out_list.append(out)
        interm_net = nn.Sequential(*list(self.net.children())[:4])
        out = interm_net(x)
        out_list.append(out)
        interm_net = nn.Sequential(*list(self.net.children())[:-2])
        out = interm_net(x)
        out_list.append(out)

        return out_list


class LeNetClassifier(Classifier):
    def __init__(self, config):
        super().__init__(config)
        num_c = 2
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_c)
        self.to(self.device)
        self.setup_optimizer()


    def forward(self, x):
        x = x.to(self.device)
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return F.log_softmax(out, dim=1)

    # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.conv1(x))
        out_list.append(out)
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(self.conv2(out))
        out_list.append(out)
        out = F.max_pool2d(out, 2, 2)
        out = out.view(-1, 320)
        out = F.relu(self.fc1(out))
        out_list.append(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1), out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        if layer_index == 0:
            out = F.relu(self.conv1(x))
        elif layer_index == 1:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2, 2)
            out = F.relu(self.conv2(out))
        elif layer_index == 2:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2, 2)
            out = out.view(-1, 320)
            out = F.relu(self.fc1(out))

        return out


class MixtureClassifier(Classifier):
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(config['x_c'] * config['x_h'] * config['x_w'], config['h1_dim'])
        self.fc2 = nn.Linear(config['h1_dim'], config['h2_dim'])

        # split into pre-defined num of experts
        self.fc_mixture = nn.ModuleList()
        for n in range(config['n_expert']):
            self.fc_mixture.append(nn.Linear(config['h2_dim'], config['y_c']))

        self.to(self.device)
        self.setup_optimizer()


    def forward(self, x, g):
        x = x.to(self.device)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc_mixture[g](out)

        return F.log_softmax(out, dim=1)

    def forward_begin(self, x, g):
        x = x.to(self.device)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc_mixture[g](out)

        return F.log_softmax(out, dim=1)

    def forward_all(self, x):
        x = x.to(self.device).view(x.size(0), -1)
        out = F.relu(self.fc1(x))
        shared_feat = F.relu(self.fc2(out))

        mixture_out = []
        log_softmax_outs = []
        for n in range(self.config['n_expert']):
            out = self.fc_mixture[n](shared_feat)
            log_softmax_outs.append(F.log_softmax(out, dim=1))

            mixture_out.append(out)

        return log_softmax_outs

    def feature_list(self, x):
        out_list = []
        x = x.to(self.device)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1), out_list

    # function to extract a specific feature
    def intermediate_forward(self, x, layer_index):
        if layer_index == 0:
            out = F.relu(self.conv1(x))
        elif layer_index == 1:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2, 2)
            out = F.relu(self.conv2(out))
        elif layer_index == 2:
            out = F.relu(self.conv1(x))
            out = F.max_pool2d(out, 2, 2)
            out = F.relu(self.conv2(out))
            out = F.max_pool2d(out, 2, 2)
            out = out.view(-1, 320)
            out = F.relu(self.fc1(out))

        return out

    def nll_all(self, x, y, step=None):
        y = y.to(self.device)
        log_softmaxs = self.forward_all(x)
        loss_preds = []
        for log_softmax in log_softmaxs:
            loss_preds.append(self.ce_loss(log_softmax, y))

        return loss_preds
