from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
import os
from generator import StyleGenerator
from discriminator import StyleDiscriminator,StyleDiscriminator_newloss
from PIL import Image
from utils import *
import matplotlib as mpl
import pickle
mpl.use('Agg')
import matplotlib.pyplot as plt
from myDataLoader import *
import numpy as np
plt.style.use('bmh')
parser = argparse.ArgumentParser(description='face attack implementation')
parser.add_argument('--model_g_path',default='/media/dsg3/FaceAttack/model/', help='save path of generator')
parser.add_argument('--enable_new_loss',default='True', help='save path of generator')

parser.add_argument('--target_face_path', default="", help='target attack face path')
parser.add_argument('--target_label_path', default="", help='target attack face label path')
parser.add_argument('--train_face_path', default="", help='training dataset path')
parser.add_argument('--train_face_label_path', default="", help='training dataset label path')
args = parser.parse_args()
def t2np(stick):
    stick = stick * 0.5+0.5
    stick = np.array(stick)
    stick = stick.transpose(1, 2, 0)
    stick = stick * 255
    img = Image.fromarray(stick.astype('uint8'))
    return img
from torch.autograd import Variable
import copy
def visualize(G, img_path, target_path,x,y):
    G.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #nowimg = Image.open(img_path)

    face = img_path
    with open('../dataset/doodle.p','rb') as f:
        patch = pickle.load(f)
    patch = patch[0]
    patch = (patch-0.5) / 0.5
    face = face.unsqueeze(0)
    patch = patch.unsqueeze(0)
    stick = stick_patch_on_face(copy.deepcopy(face),patch,x,y)
    stick = stick.squeeze(0)
    img = t2np(stick)
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)
    #target_img = Image.open(target_path)
    target = target_path
    target=target.unsqueeze(0)
    adv = G(target).detach().cpu()
    stick_adv = stick_patch_on_face(copy.deepcopy(face),adv,x,y)
    stick_adv = stick_adv.squeeze(0)
    img_adv = t2np(stick_adv)
    plt.subplot(2,3,2)
    plt.imshow(img_adv)

    patch = patch.squeeze(0)
    adv = adv.squeeze(0)
    patch_img = t2np(patch)
    adv_img = t2np(adv)
    plt.subplot(2,3,4)
    plt.imshow(patch_img)
    plt.subplot(2,3,5)
    plt.imshow(adv_img)

    patch_opt = Image.open('../nowresult.jpg').convert('RGB')

    plt.subplot(2,3,6)
    plt.imshow(patch_opt)
    patch_opt = transform(patch_opt).unsqueeze(0)
    stick_opt = stick_patch_on_face(copy.deepcopy(face), patch_opt,x,y)
    stick_opt = t2np(stick_opt.squeeze(0))
    
    plt.subplot(2,3,3)
    plt.imshow(stick_opt)

    plt.savefig('./result.jpg')
def visualize_softmax(G, img_path, target_path,x,y):
    G.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #nowimg = Image.open(img_path)

    face = img_path
    with open('../dataset/doodle.p','rb') as f:
        patch = pickle.load(f)
    patch = patch[0]
    patch = (patch-0.5) / 0.5
    face = face.unsqueeze(0)
    patch = patch.unsqueeze(0)
    stick = stick_patch_on_face(copy.deepcopy(face),patch,x,y)
    stick = stick.squeeze(0)
    img = t2np(stick)
    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)
    #target_img = Image.open(target_path)
    target = target_path
    target=target.unsqueeze(0)
    adv = G(target).detach().cpu()
    stick_adv = stick_patch_on_face(copy.deepcopy(face),adv,x,y)
    stick_adv = stick_adv.squeeze(0)
    img_adv = t2np(stick_adv)
    plt.subplot(2,3,2)
    plt.imshow(img_adv)

    patch = patch.squeeze(0)
    adv = adv.squeeze(0)
    patch_img = t2np(patch)
    adv_img = t2np(adv)
    plt.subplot(2,3,4)
    plt.imshow(patch_img)
    plt.subplot(2,3,5)
    plt.imshow(adv_img)

    patch_opt = Image.open('../nowresult_softmax.jpg').convert('RGB')

    plt.subplot(2,3,6)
    plt.imshow(patch_opt)
    patch_opt = transform(patch_opt).unsqueeze(0)
    stick_opt = stick_patch_on_face(copy.deepcopy(face), patch_opt,x,y)
    stick_opt = t2np(stick_opt.squeeze(0))
    
    plt.subplot(2,3,3)
    plt.imshow(stick_opt)

    plt.savefig('./result_softmax.jpg')
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset, testset = load_file(args.train_face_label_path, 'CASIA')
    face_train_dataset = Train_Dataset(args.train_face_path, trainset, transform = transform)

    trainset, testset = load_file(args.train_face_label_path, 'CASIA', test=True)
    face_target_dataset = Train_Dataset(args.target_face_path, trainset, transform = transform)

    target_face_loader = DataLoader(dataset = face_target_dataset, batch_size = 1, shuffle = True, drop_last = False)
    print('Total length of target face set: ', target_face_loader.__len__())
    face_train_loader = DataLoader(dataset = face_train_dataset, batch_size = 1, shuffle = True, drop_last = False)

    print('load train set')
    print('Total length of train face set: ', face_train_loader.__len__())
    for i,(face, label, xx, yy) in enumerate(target_face_loader):
        if(i==0):
            nowtarget = face.squeeze(0)
            break
    for i,(face, label, xx, yy) in enumerate(face_train_loader):
        if(i==1):
            nowtrain = face.squeeze(0)
            x=xx
            y=yy
            break
    G = StyleGenerator()
    G_soft = StyleGenerator()
    if(args.enable_new_loss=='True'):
          G.load_state_dict(torch.load(args.model_g_path+'faceAttack_G_newloss.pkl',map_location='cpu'))
          G_soft.load_state_dict(torch.load(args.model_g_path+'faceAttack_G_newloss_softmax.pkl',map_location='cpu'))
    else:
        G.load_state_dict(torch.load(args.model_g_path+'faceAttack_G.pkl',map_location='cpu'))
    print(x,y)
    visualize(G, nowtrain,nowtarget,y,x)
    visualize_softmax(G_soft, nowtrain,nowtarget,y,x)