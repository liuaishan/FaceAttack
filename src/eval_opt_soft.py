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

from Loss.CosineFace import AddMarginProduct
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
def getmax(num_list,topk=3):
    tmp_list=copy.deepcopy(num_list)
    tmp_list.sort()
    #min_num_index=[num_list.index(one) for one in tmp_list[:topk]]
    max_num_index=[num_list.index(one) for one in tmp_list[::-1][:topk]]
    return max_num_index
def test_op(G, model, target_face_id, train_face_id, patch_num, metric):
    G = G.cuda()
    G.eval()
    metric = metric.cuda()
    metric.eval()
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
    #with open('../dataset/doodle_small.p','rb') as f:
        #patchset=pickle.load(f)
    #randomly find original patch
    #patch_ori = patchset[0]
    #patch_ori = (patch_ori-0.5) / 0.5
    from PIL import Image
    patch_ori = Image.open('1.jpg').convert('RGB')
    patch_ori = transform(patch_ori).unsqueeze(0).cuda()
    #512 train
    trainset, testset = load_file(args.test_label_path, args.test_dataset)
    trainset_test, testset_test = load_file_test(args.test_label_path, args.test_dataset)
    trainset1 = [item for item in trainset_test if item not in trainset]
    print('length of file', len(trainset),len(trainset_test))
    face_test_dataset = Train_Dataset(args.test_face_path, trainset1, transform = transform)
    face_test_loader = DataLoader(dataset = face_test_dataset, batch_size = 1, shuffle = False, drop_last = False)
    print('length of face test loader',face_test_loader.__len__(), len(trainset1))
    #96 target
    trainset, testset = load_file(args.target_label_path, args.test_dataset, test=True)
    trainset_test, testset_test = load_file_test(args.target_label_path, args.test_dataset, test=True)
    trainset1 = [item for item in trainset_test if item not in trainset]
    print('length of file', len(trainset),len(trainset_test))
    face_target_dataset = Train_Dataset(args.target_face_path, trainset1, transform = transform)
    target_face_loader = DataLoader(dataset = face_target_dataset, batch_size = 1, shuffle = False, drop_last = False)
    print('length of face target loader',target_face_loader.__len__(), len(trainset1))
    #randomly find target face to generate patch
    '''
    for i,(face, label) in enumerate(target_face_loader):
        if(i==target_face_id):
            nowface = face#Variable(face).cuda()
            break
    nowpatch = G(nowface).cuda()
    '''
    #randomly find the face to be tested
    for i,(face, label,x,y) in enumerate(face_test_loader):
        if(i==train_face_id):
            testface = Variable(face).cuda()
            nowx=x
            nowy=y
            break
    import copy
    target_label = torch.Tensor([46]).long().cuda() 
    mindis = []
    '''
    for i,(face, label) in enumerate(face_test_loader):
        #face = Variable(face).cuda()
        distance = predict(model, testface,face)
        mindis.append(distance.item())
    '''
    nowtot=0
    correct = 0
    total = 0
    for i,(face, label,aa,bb) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        label = Variable(label.cuda().long())
        adv_logit = model(face)
        output = metric(adv_logit,label)
        _, predicted = torch.max(output.data, 1)#得到每行输出的最大值，即最大概率的分类结果，一共两列，第一列是原始数据，第二列是预测结果，要第二列
        total += label.size(0)
        correct += (predicted == target_label).sum().item()
    print('original pic',100.0 * correct/total)
    #patch_ori1 = patch_ori.unsqueeze(0)
    nowtot = 0
    #face_oripatch = stick_patch_on_face(copy.deepcopy(testface), (patch_ori),nowy,nowx).cuda()
    mindis = []
    '''
    for i,(face, label) in enumerate(face_test_loader):
        #face = Variable(face).cuda()
        distance = predict(model, face_oripatch,face)
        mindis.append(distance.item())
    '''
    correct = 0
    total = 0
    for i,(face, label,aa,bb) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        label = Variable(label.cuda().long())
        face_oripatch = stick_patch_on_face(copy.deepcopy(face),patch_ori, bb,aa).cuda()
        #distance = predict(model, face_oripatch, face)
        adv_logit = model(face_oripatch)
        output = metric(adv_logit,label)
        _, predicted = torch.max(output.data, 1)#得到每行输出的最大值，即最大概率的分类结果，一共两列，第一列是原始数据，第二列是预测结果，要第二列
        total += label.size(0)
        correct += (predicted == target_label).sum().item()
    print('with original patch',100.0 * correct/total)

    #adv_face = stick_patch_on_face(testface, nowpatch).cuda()
    mindis = []
    nowtot=0
    adv_patch = Image.open('../nowresult_softmax.jpg').convert('RGB')
    adv_patch = transform(adv_patch).unsqueeze(0).cuda()
    correct = 0
    total = 0
    for i,(face, label,aa,bb) in enumerate(face_test_loader):
        face = Variable(face).cuda()
        label = Variable(label.cuda().long())
        #nowpatch = G(face)
        adv_face = stick_patch_on_face(copy.deepcopy(face), adv_patch,bb,aa).cuda()
        adv_logit = model(adv_face)
        output = metric(adv_logit,label)
        _, predicted = torch.max(output.data, 1)#得到每行输出的最大值，即最大概率的分类结果，一共两列，第一列是原始数据，第二列是预测结果，要第二列
        total += label.size(0)
        correct += (predicted == target_label).sum().item()
    print('with adv patch',100.0 * correct/total)

def choose_model():
    if args.model == 'se_resnet_50':
        metric_fc = AddMarginProduct(512, 10575, s=30, m=0.35)
        sub_model = SEResNet_IR(50, mode='se_ir')
        sub_model.load_state_dict(torch.load(args.model_path))
        metric_fc.load_state_dict(torch.load('./margin_res50IR_cos_CA.pkl'))
    elif args.model == 'resnet18':
        pass
    #sub_model.cuda()
    return sub_model,metric_fc

if __name__ == "__main__":
    print()
    print('now using softmax and optimization')
    print()
    cnn,metric = choose_model()
    G = StyleGenerator()
    #G.load_state_dict(torch.load(args.model_g_path+'faceAttack_G_newloss.pkl'))
    test_op(G,cnn,8,150,0,metric)
