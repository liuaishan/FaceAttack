from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from Model.SE_ResNet_IR import *
from lfw_DataLoader import *
from torch.optim.lr_scheduler import StepLR
import os
from myDataLoader import *
from lfw_test import *
from generator import StyleGenerator
from discriminator import StyleDiscriminator

from utils import *
import matplotlib as mpl

#mpl.use('Agg')
import matplotlib.pyplot as plt

#plt.style.use('bmh')

from torch.autograd import Variable
import pickle
parser = argparse.ArgumentParser(description='face attack implementation')
parser.add_argument('--model', default="se_resnet_50", help='number of classes')
parser.add_argument('--model_path', default="", help='number of classes')
parser.add_argument('--model_g_path',default='', help='save path of generator')
parser.add_argument('--test_face_path', default="", help='test data path')
parser.add_argument('--test_dataset', default="CASIA", help='type of test dataset')
parser.add_argument('--test_label_path', default="", help='test label path')
parser.add_argument('--target_face_path', default="", help='target attack face path')
parser.add_argument('--target_label_path', default="", help='target attack face label path')
args = parser.parse_args()
def get_bool(string):
    if (string == 'False'):
        return False
    else:
        return True
import heapq
import copy
def getmin(num_list,topk=3):
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    min_num_index=[num_list.index(one) for one in tmp_list[:topk]]
    return min_num_index
def test_op(G, model):
    G = load_model(args.model_g_path+'faceAttack_G.pkl')
    G = G.cuda()
    G.eval()
    model=model.cuda()
    model.eval()
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count()," GPUs")
        totdev=torch.cuda.device_count()
        G = nn.DataParallel(G)
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("1 Gpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    with open('../dataset/doodle_small.p','rb') as f:
        patchset=pickle.load(f)
    patch_ori = patchset[0]
    patch_ori = (patch_ori-0.5) / 0.5
    trainset, testset = load_file(args.test_label_path, args.test_dataset)
    face_test_dataset = Train_Dataset(args.test_face_path, trainset, transform = transform)
    face_test_loader = DataLoader(dataset = face_test_dataset, batch_size = 1, shuffle = False, drop_last = False)
    print('length of face test loader',face_test_loader.__len__())
    for i,(face, label) in enumerate(face_test_loader):
        if(i==0):
            face = Variable(face).cuda()
    nowface = face
    trainset, testset = load_file(args.target_label_path, args.test_dataset, test=True)
    face_target_dataset = Train_Dataset(args.target_face_path, trainset, transform = transform)
    target_face_loader = DataLoader(dataset = face_target_dataset, batch_size = 1, shuffle = False, drop_last = False)
    print('length of face target loader',target_face_loader.__len__())
    mindis = []
    for i,(face, label) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        distance = predict(model, nowface,face)
        mindis.append(distance)
    for i,(face, label) in enumerate(target_face_loader):
        face = Variable(face).cuda()
        if(i==50):
            nowpatch = G(face)
        distance = predict(model, nowface,face)
        mindis.append(distance)
    minlist = getmin(mindis, topk=10)
    print('original pic',minlist)
    face_oripatch = stick_patch_on_face(nowface, patch_ori)
    mindis = []
    for i,(face, label) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        distance = predict(model, face_oripatch,face)
        mindis.append(distance)
    for i,(face, label) in enumerate(target_face_loader):
        face = Variable(face).cuda()
        if(i==50):
            nowpatch = G(face)
        distance = predict(model, face_oripatch,face)
        mindis.append(distance)
    minlist = getmin(mindis, topk=10)
    print('with original patch',minlist)
    adv_face = stick_patch_on_face(nowface, nowpatch)
    mindis = []
    for i,(face, label) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        distance = predict(model, adv_face,face)
        mindis.append(distance)
    for i,(face, label) in enumerate(target_face_loader):
        face = Variable(face).cuda()
        if(i==50):
            nowpatch = G(face)
        distance = predict(model, adv_face,face)
        mindis.append(distance)
    minlist = getmin(mindis, topk=10)
    print('with adv patch',minlist)
def choose_model():
    if args.model == 'se_resnet_50':
        sub_model = SEResNet_IR(50, mode='se_ir')
        sub_model.load_state_dict(torch.load(args.model_path))
    elif args.model == 'resnet18':
        pass
    sub_model.cuda()
    return sub_model

def load_model(g_path=None, d_path=None):
    G = StyleGenerator()
    if os.path.exists(g_path) == False:
        print('Load Generator failed')
    else:
        print('Successfully load G')
        G.load_state_dict(torch.load(g_path))
    return G


if __name__ == "__main__":
    cnn = choose_model()
    
    test_op(G,cnn)
