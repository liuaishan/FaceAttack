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
#一些超参数
learning_rate=0.001
batch_size=64
epoch_num=6666
#数据集图片文件夹
root='/media/dsg3/datasets/CASIA_WebFace/CASIA_align/'
#数据集标签文件
datafile='/home/dsg/xuyitao/Face/FaceTrain/CASIA_list.txt'
name_model='Res'

trainset=[]
valset=[]
testset=[]
fh = open(datafile, 'r')
failed = open('/home/dsg/xuyitao/Face/FaceTrain/CASIA_fail.txt','r')
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
# In[14]:


#制作数据集dataset

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
# In[4]:
'''
LOSS Part
Temporarily using 'AddMarginProduct' as our modification to CrossEntropy Loss
'''
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=14.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

class AddMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine, device = 'cuda')
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = self.s * (cosine - one_hot * self.m)

        return output


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output
'''
Model Part
Using SE-ResNet-IR50 model
'''

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
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (7, 7), stride = 2, padding = 3, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                          nn.Dropout(drop_ratio),
                                          Flatten(),
                                          nn.Linear(512 * 8 * 8, feature_dim),
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
# In[11]:

'''
Train part
'''
import os
def adjust(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
transform = transforms.Compose([
                         #transforms.Resize((112,112)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                     ])
class Train:
    def __init__(self, face,transform,trset,vaset,teset,root,model_name, datafile, number_classes = 10177, model_path = '/home/dsg/xuyitao/Face/FaceTrain/model/margin_res50IR_cos_CA.pkl', para_path = '/home/dsg/xuyitao/Face/FaceTrain/model/params_res50IR_cos_CA.pkl',pretrain=False,batch_size=64):
        #sammodel = models.resnet18(pretrained=False)
        #sammodel.load_state_dict(torch.load('/userhome/data/CelebA_crop/resnet18-5c106cde.pth'))
        self.model=SEResNet_IR(50, mode='se_ir')
        #self.model.load_state_dict(torch.load('/userhome/data/CelebA_crop/model/params_res_cos.pkl'))
        
        
        self.batch_size = batch_size
        self.train_data=CelebA(dataset=trset,root=root,datatxt='datafile', transform=transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True,num_workers=4,drop_last=True)
        
        #self.val_data=CelebA(dataset=vaset,root=root,datatxt='datafile', transform=transform)
        #self.val_loader = torch.utils.data.DataLoader(dataset=self.val_data, batch_size=batch_size, shuffle=True)
        
        self.test_data=CelebA(dataset=teset,root=root,datatxt='datafile', transform=transform)
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
        
        #self.model.load_state_dict(torch.load('/userhome/data/CelebA_crop/model/params_res50IR_cos_CA.pkl'))
        #self.metric_fc.load_state_dict(torch.load('/userhome/data/CelebA_crop/model/margin_res50IR_cos_CA.pkl'))


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
                    
        


# In[17]:


#测试能否开始训练
train=Train('cosine',transform, trainset,valset,testset,root=root,model_name=name_model,datafile=datafile,batch_size=batch_size)
print(train.batch_size)
#train.start_train()


# In[18]:


#测试能否开始训练
train.start_train(epoch=epoch_num,learning_rate=learning_rate)



# In[ ]:




