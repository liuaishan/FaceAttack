import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


class BottleNeck_IR(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck_IR, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return shortcut + res

class BottleNeck_IR_SE(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeck_IR_SE, self).__init__()
        if in_channel == out_channel:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_channel),
                                       nn.Conv2d(in_channel, out_channel, (3, 3), 1, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.PReLU(out_channel),
                                       nn.Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       SEModule(out_channel, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return shortcut + res


class Bottleneck(namedtuple('Block', ['in_channel', 'out_channel', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, out_channel, num_units, stride=2):
    return [Bottleneck(in_channel, out_channel, stride)] + [Bottleneck(out_channel, out_channel, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=4),
            get_block(in_channel=128, out_channel=256, num_units=14),
            get_block(in_channel=256, out_channel=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=13),
            get_block(in_channel=128, out_channel=256, num_units=30),
            get_block(in_channel=256, out_channel=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, out_channel=64, num_units=3),
            get_block(in_channel=64, out_channel=128, num_units=8),
            get_block(in_channel=128, out_channel=256, num_units=36),
            get_block(in_channel=256, out_channel=512, num_units=3)
        ]
    return blocks


class SEResNet_IR(nn.Module):
    def __init__(self, num_layers, feature_dim=512, drop_ratio=0.4, mode = 'ir'):
        super(SEResNet_IR, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'se_ir'], 'mode should be ir or se_ir'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleNeck_IR
        elif mode == 'se_ir':
            unit_module = BottleNeck_IR_SE
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 7 * 7, feature_dim),
                                          nn.BatchNorm1d(feature_dim))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.out_channel,
                                bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x