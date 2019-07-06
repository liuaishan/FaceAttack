from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from collections import namedtuple
import torch.utils.data as data
import os
from utils import *
device = torch.device("cuda")
cudnn.benchmark = True
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


def extractDeepFeature(img, model, is_gray):
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            #transforms.Resize((112,112)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    #img, img_ = transform(img), transform(F.hflip(img))
    img, img_ = transform(img), transform(img)
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    out1 = model(img)
    out2 = model(img_)
    ft = torch.cat((out1, out2), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        try:
            test = base[i * n // n_folds:(i + 1) * n // n_folds]
            train = list(set(base) - set(test))
            folds.append([train, test])
        except:
            print("fail in Kfold:")
            print(i)
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy
def extractDeepFeature_predict(img, model, is_gray=False):
    #img, img_ = img.to('cuda'), img.to('cuda')
    out1 = model(img)
    #out2 = model(img_)
    #ft = torch.cat((out1, out2), 1)[0].to('cpu')
    return out1.cpu()

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold
def predict(model, targetface, trainface , best_threshold=0.3001):
    f1 = extractDeepFeature_predict(targetface, model)
    f2 = extractDeepFeature_predict(trainface, model)
    for i in range(targetface.size()[0]):
        if(i==0):
            distance = f1[i].dot(f2[i]) / (f1[i].norm() * f2[i].norm() + 1e-5)
        else:
            distance = distance + f1[i].dot(f2[i]) / (f1[i].norm() * f2[i].norm() + 1e-5)
    return distance / targetface.size()[0]
def eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose([
            #transforms.Resize((112,112)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    root = '/media/dsg3/datasets/lfw/lfw_align/'
    with open('/media/dsg3/datasets/lfw/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]
    patch = Image.open('./ori_patch.jpg').convert('RGB')
    patch = transform(patch).unsqueeze(0).cuda()
    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if(i % 100 == 0 and i > 0):
                print(i,distance ,distance1,predict(model, img1, img2))
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model, is_gray)

            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            img1 = transform(img1).unsqueeze(0).cuda()
            img2 = transform(img2).unsqueeze(0).cuda()

            f11 = extractDeepFeature_predict(img1,model)[0]
            img2 = stick_patch_on_face(img2,patch).cuda()
            f12 = extractDeepFeature_predict(img2,model)[0]
            distance1=f11.dot(f12) / (f11.norm() * f12.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.001)
    new_pred = []
    #new_pred = np.array(map(lambda line: line.strip('\n').split(), predicts))
    for i in predicts:
        new_pred.append(i.strip('\n').split('\t'))
    predicts = new_pred
    predicts = np.array(predicts)
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts
def test_eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '/media/dsg3/datasets/lfw/lfw_align/'
    with open('/media/dsg3/datasets/lfw/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if(i % 100 == 0 and i > 0):
                print(i,distance)
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model, is_gray)
            f2 = extractDeepFeature(img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            print(distance)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))
            break
    return 0
def img_loader(path):
    try:
        with open(path, 'rb') as f:
            #img = cv2.imread(path)
            img = Image.open(path)
            return img
    except IOError:
        print('Cannot load image ' + path)

class LFW(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                fold = i // 600
                flag = 1
            elif len(p) == 4:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, img_r]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)
if __name__ == '__main__':
    #model=SEResNet_IR(50, mode='se_ir')
    #model_class = Resnet50FaceModel
    model=SEResNet_IR(50, mode='se_ir')
    #_, result = eval(model.to('cuda'), model_path='./model/params_res50IR_cos_CA.pkl')
    _, result = eval(model.to('cuda'), model_path='/media/dsg3/xuyitao/Face/model/params_res50IR_cos_CA.pkl')
    #test_eval(model.to('cuda'), model_path='/media/dsg3/xuyitao/Face/model/params_res50IR_cos_CA.pkl')
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = LFW('/media/dsg3/datasets/lfw/lfw_align/', '/media/dsg3/datasets/lfw/pairs.txt', transform=transform)
    #dataset = LFW(root, file_list)
    trainloader = data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for img in trainloader:
        f1 = extractDeepFeature_predict(img[1],model)
        f2 = extractDeepFeature_predict(img[0],model)
        distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
        print(distance)
        break
    '''
    pass