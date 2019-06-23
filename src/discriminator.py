import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_
from StyleUtils import *


class StyleDiscriminator(nn.Module):
    def __init__(self,
                 resolution=64,  # 输入图片的分辨率 为 64* 64
                 fmap_base=8192,
                 num_channels=3,
                 fmap_max=512,
                 fmap_decay=1.0,
                 # f=[1, 2, 1]         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """
            判别器，支持输入宽高相等的图片。默认图片分辨率为32*32
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))   
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4 #图片分辨率必须为 2 的n 次幂，并且 不能低于4*4
        
        # 将输入的rgb图片转换为 特征图
        self.fromrgb = nn.Conv2d(num_channels, 512, kernel_size=1)
        # blur2d
        self.blur2d = Blur2d(f)

        # 下采样  
        self.down1 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.down2 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.down3 = nn.Conv2d(512, 512, kernel_size=2, stride=2)
        self.down4 = nn.Conv2d(512, 512, kernel_size=2, stride=2)


        # 卷积层定义
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))

        # 计算预测结果
        self.conv_last = nn.Conv2d(512, 512, kernel_size=3, padding=(1, 1))
        self.dense0 = nn.Linear(fmap_base, 512)
        self.dense1 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = F.leaky_relu(self.fromrgb(input), 0.2, inplace=True)
        # 5. 64 x 64 -> 32 x 32
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down1(self.blur2d(x)), 0.2, inplace=True)
        #32 x 32 -> 16 x 16
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down2(self.blur2d(x)), 0.2, inplace=True)
        #16 x 16 -> 8 x 8
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down3(self.blur2d(x)), 0.2, inplace=True)
        #8 x 8 -> 4 x 4
        x = F.leaky_relu(self.conv4(x), 0.2, inplace=True)
        x = F.leaky_relu(self.down4(self.blur2d(x)), 0.2, inplace=True)
        #4 x 4 -> point
        x = F.leaky_relu(self.conv_last(x), 0.2, inplace=True)
        # N x 8192
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.dense0(x), 0.2, inplace=True)
        # N x 1
        x = torch.sigmoid(self.dense1(x))#How to ensure the result is in [0,1]?
        #x = torch.sigmoid(x)
        return x
if __name__ == '__main__':
    D = StyleDiscriminator()
    x = torch.randn(1,3,64,64)
    z = D(x)
    print(z)