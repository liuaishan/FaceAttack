import os
import torch.utils.data as data
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
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
#test lfw building
if __name__ == '__main__':
    root = 'E:/lfw/lfw-112x112'
    file_list = 'E:/lfw/pairs.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = LFW(root, file_list, transform=transform)
    #dataset = LFW(root, file_list)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    print(len(dataset))
    tot=0
    for data in trainloader:
        print(len(data))
        for d in data:
            print(d[0].shape)
        tot+=1
        print('----------')
        if(tot==5):
            break