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
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#plt.style.use('bmh')
parser = argparse.ArgumentParser(description='face attack implementation')
parser.add_argument('--model_g_path',default='../', help='save path of generator')
parser.add_argument('--enable_new_loss',default='True', help='save path of generator')
args = parser.parse_args()
def t2np(stick):
    stick = stick * 0.5+0.5
    stick = np.array(stick)
    stick = stick.transpose(1, 2, 0)
    stick = stick * 255
    img = Image.fromarray(stick.astype('uint8'))
    return img
from torch.autograd import Variable
import random
def visualize(G, img_path, num):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nowimg = Image.open(img_path)
    face = transform(nowimg)
    with open('../dataset/doodle.p','rb') as f:
        patch = pickle.load(f)
    idx = random.randint(0,31)
    patch = patch[idx]
    patch = (patch-0.5) / 0.5
    face = face.unsqueeze(0)
    patch = patch.unsqueeze(0)
    stick = stick_patch_on_face(face,patch)
    stick = stick.squeeze(0)
    img = t2np(stick)
    plt.figure()
    plt.subplot(221)
    plt.imshow(img)

    target = transform(nowimg)
    target=target.unsqueeze(0)
    adv = G(target).detach().cpu()
    stick_adv = stick_patch_on_face(face,adv)
    stick_adv = stick_adv.squeeze(0)
    img_adv = t2np(stick_adv)
    plt.subplot(222)
    plt.imshow(img_adv)

    patch = patch.squeeze(0)
    adv = adv.squeeze(0)
    patch_img = t2np(patch)
    adv_img = t2np(adv)
    plt.subplot(223)
    plt.imshow(patch_img)
    plt.subplot(224)
    plt.imshow(adv_img)

    plt.savefig('./result'+str(num)+'.jpg')
if __name__ == "__main__":
    G = StyleGenerator()
    if(args.enable_new_loss=='True'):
        G.load_state_dict(torch.load(args.model_g_path+'faceAttack_G_newloss.pkl',map_location='cpu'))
    else:
        G.load_state_dict(torch.load(args.model_g_path+'faceAttack_G.pkl',map_location='cpu'))
    visualize(G, './063.jpg',0)