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
from Loss.CosineFace import *
from Model.SE_ResNet_IR import *
from Dataset.CASIA import *
from Dataset.CelebA import *
import argparse
parser = argparse.ArgumentParser(description='model interpretation')
parser.add_argument('--load_model_path', default="'/home/dsg/xuyitao/Face/FaceTrain/model/params_res50IR_cos_CA.pkl'", help='load model path')
parser.add_argument('--load_margin_path', default="'/home/dsg/xuyitao/Face/FaceTrain/model/margin_res50IR_cos_CA.pkl'", help='load model path')
parser.add_argument('--save_model_path', default="'/home/dsg/xuyitao/Face/FaceTrain/model/params_res50IR_cos_CA.pkl'", help='save model path')
parser.add_argument('--save_margin_path', default="'/home/dsg/xuyitao/Face/FaceTrain/model/margin_res50IR_cos_CA.pkl'", help='save model path')
parser.add_argument('--train_data_root', default="/media/dsg3/datasets/CASIA_WebFace/CASIA_align/", help='training dataset root')
parser.add_argument('--train_data_list', default="/home/dsg/xuyitao/Face/FaceTrain/CASIA_list.txt", help='training dataset file list')
parser.add_argument('--train_data_fail', default="/home/dsg/xuyitao/Face/FaceTrain/CASIA_list.txt", help='training dataset failed list')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--train_dataset', default='CASIA', default=0.0002, help='type of train dataset')
parser.add_argument('--batchsize', type=int, default=64, help='training batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
args = parser.parse_args()
#一些超参数
learning_rate=args.lr
batch_size=args.batchsize
epoch_num=args.epoch
#数据集图片文件夹
root=args.train_data_root
#数据集标签文件
datafile=args.train_data_list
name_model='Res'

trainset=[]
valset=[]
testset=[]
fh = open(datafile, 'r')
failed = open(args.train_data_fail,'r')
fail=[]
k=0
m=0
for line in failed:
    line = line.rstrip()       #删除本行末尾的空格
    words = line.split() 
    fail.append(words[1])
for line in fh:                #按行循环txt文本中的内容
    line = line.rstrip()       #删除本行末尾的空格
    words = line.split()       #通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
    if(k<9099):
        if(words[0]==fail[k]):
            k+=1
            continue
    if(m%100==4):
        testset.append((words[0],int(words[1])))
    else:
        trainset.append((words[0],int(words[1])))
    m=m+1

'''
Train part
'''
import os
def adjust(optimizer, epoch):
    lr = 0.01 * (epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
transform = transforms.Compose([
                         #transforms.Resize((112,112)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                     ])
class Train:
    def __init__(self, face,transform,trset,vaset,teset,root,model_name, datafile, number_classes = 10177, model_path = args.save_margin_path, para_path = args.save_model_path,pretrain=False,batch_size=64):
        #sammodel = models.resnet18(pretrained=False)
        #sammodel.load_state_dict(torch.load('/userhome/data/CelebA_crop/resnet18-5c106cde.pth'))
        self.model=SEResNet_IR(50, mode='se_ir')
        #self.model.load_state_dict(torch.load('/userhome/data/CelebA_crop/model/params_res_cos.pkl'))
        
        
        self.batch_size = batch_size
        self.train_data=CASIA(dataset=trset,root=root,datatxt='datafile', transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True,num_workers=4,drop_last=True)
        
        #self.val_data=CelebA(dataset=vaset,root=root,datatxt='datafile', transform=transform)
        #self.val_loader = torch.utils.data.DataLoader(dataset=self.val_data, batch_size=batch_size, shuffle=True)
        
        self.test_data=CASIA(dataset=teset,root=root,datatxt='datafile', transform=transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=4, shuffle=True,num_workers=4,drop_last=True)
        
        print(self.train_data.__len__(),self.test_data.__len__())
        if face == 'cosine':
            self.metric_fc = AddMarginProduct(512, 10575, s=30, m=0.35)
        elif face == 'arcface':
            self.metric_fc = ArcMarginProduct(512, 10575, s=30, m=0.5, easy_margin=False)
        elif face == 'sphereface':
            self.metric_fc = SphereProduct(512, 10575, m=4)
        else:
            self.metric_fc = nn.Linear(512, 10575)
        
        #self.model.load_state_dict(torch.load(args.load_model_path))
        #self.metric_fc.load_state_dict(torch.load(args.load_margin_path))


        self.model_path=model_path
        self.para_path=para_path
        self.totdev=1
        if torch.cuda.is_available():
            self.model.cuda()
            self.metric_fc.cuda()
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count()," GPUs")
            self.totdev=torch.cuda.device_count()
            self.metric_fc = nn.DataParallel(self.metric_fc)
            self.model = nn.DataParallel(self.model)
        elif torch.cuda.device_count() == 1:
            print("1 Gpu")
        else:
            print("Only use CPU")

        #if torch.cuda.is_available():
           # self.model.cuda()

    def start_train(self, epoch=1111,learning_rate=0.0001, batch_display=50):
        self.epoch_num = epoch
        self.lr = learning_rate
        if torch.cuda.is_available():
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD([{'params': self.model.parameters(),'weight_decay':5e-4}, {'params': self.metric_fc.parameters(),'weight_decay':5e-4}],
                                lr=0.01,
                                momentum=0.9,
                                nesterov=True)

        totad = 0
        jd = 1
        self.model.train()
        for epoch in range(self.epoch_num):
            for i, (images, labels) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    input_images = Variable(images.cuda())
                    target_labels = Variable(labels.cuda().long())
                else:
                    input_images = Variable(images)
                    target_labels = Variable(labels)
                
                try:
                # 正向传播算结果
                    feature = self.model(input_images)
                    outputs = self.metric_fc(feature,target_labels)
                    #outputs = self.model(input_images)
                    loss = criterion(outputs, target_labels)

                    # 反向传播算参数调整
                    optimizer.zero_grad()#对每批数据，梯度都要清零
                    loss.backward()#求梯度
                    optimizer.step()#梯度下降走一步
                    if(loss<=1 and jd==1):
                            totad = totad + 1
                            totnum = 0.995**totad
                            adjust(optimizer,totnum)
                            if(totad >= 900):
                                jd=0
                    if (i % batch_display == 0):
                        pred_prob, pred_label = torch.max(outputs, dim=1)
                        print("Input Label : ", target_labels[:6])
                        print("Output Label : ", pred_label[:6])
                        batch_correct = (pred_label == target_labels).sum().item() * 1.0 / self.batch_size
                        print("Epoch : %d, Batch : %d, Loss : %f, Batch Accuracy %f" %(epoch, i, loss, batch_correct))

                    if (epoch % 5 == 0 and i % 2500 == 0):
                        if self.totdev > 1:
                            torch.save(self.model.module.state_dict(),self.para_path)
                            torch.save(self.metric_fc.module.state_dict(),self.model_path)
                        else:
                            torch.save(self.model,self.model_path)
                            torch.save(self.model.state_dict(),self.para_path)
                        if(epoch > 10):
                            self.model.eval()
                            print("start to validate")
                            correct = 0
                            total = 0
                            for images, labels in self.test_loader:
                                images = images.cuda()
                                labels = labels.cuda().long()
                                #outputs = self.model(images)
                                fee  = self.model(images)
                                outputs = self.metric_fc(fee,labels)
                                _, predicted = torch.max(outputs.data, 1)#得到每行输出的最大值，即最大概率的分类结果，一共两列，第一列是原始数据，第二列是预测结果，要第二列
                                total += labels.size(0)
                                #print(labels.size(-1))
                                correct += (predicted == labels).sum().item()
                            self.model.train()
                            print('Validation Accuracy : {} %'.format(100 * correct / total))
                except:
                    print('ERROR TRAINING, SAVING THE MODEL')
                    if self.totdev > 1:
                        torch.save(self.model.module.state_dict(),self.para_path)
                        torch.save(self.metric_fc.module.state_dict(),self.model_path)
                    else:
                        torch.save(self.model.state_dict(),self.para_path)
                        torch.save(self.metric_fc.state_dict(),self.model_path)
                    
train=Train('cosine',transform, trainset,valset,testset,root=root,model_name=name_model,datafile=datafile,batch_size=batch_size)
print(train.batch_size)
train.start_train(epoch=epoch_num,learning_rate=learning_rate)
