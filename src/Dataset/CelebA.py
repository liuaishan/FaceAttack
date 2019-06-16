#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.nn import Parameter
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
import math
from torchvision import models
from collections import OrderedDict
from collections import namedtuple

class CelebA(torch.utils.data.Dataset): #创建自己的类：CelebA,继承torch.utils.data.Dataset
    def __init__(self,root, datatxt,dataset ,transform=None, target_transform=None): #初始化一些需要传入的参数
        self.imgs = dataset
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):    #必须要有，用于按照索引读取每个元素的具体内容，训练时是每个batch的内容
        fn, label = self.imgs[index] #fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        try:
            img = Image.open(root+fn).convert('RGB')  #按照path读入图片from PIL import Image # 按照路径读取图片
            if self.transform is not None:
                img = self.transform(img)  #是否进行transform
        except:
            print("failure %s"%fn)
            return None,None
        return img,label  #return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self):  #必须要写，返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

#train_data=CelebA(root='E:\\CelebA\\CelebA_cropimg\\',datatxt='E:\\CelebA\\Anno\\identity_CelebA.txt' ,transform=transforms.ToTensor())
#test_data=CelebA(root=root,datatxt=datafile+'test.txt', transform=transforms.ToTensor())

#train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)
#在shuffle为False情况下打印，顺序同txt
'''
for batch_index, batch in enumerate(train_loader):
    name, label = batch
    print(label)
'''