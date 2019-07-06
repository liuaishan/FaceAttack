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
import numpy as np
from myDataLoader import *
from lfw_test import *
from generator import StyleGenerator
from discriminator import StyleDiscriminator,StyleDiscriminator_newloss
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,6,7"
from utils import *
import matplotlib as mpl
from loss import *
from opt import *
from PIL import Image
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')

from torch.autograd import Variable


def get_bool(string):
    if (string == 'False'):
        return False
    else:
        return True


# Training settings
parser = argparse.ArgumentParser(description='face attack implementation')
parser.add_argument('--face_batchsize', type=int, default=1, help='training face batch size')
parser.add_argument('--patch_batchsize', type=int, default=64, help='training patch batch size')
parser.add_argument('--epoch', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--test_flag', type=get_bool, default=True, help='test or train')
parser.add_argument('--test_dataset', default="lfw", help='test dataset type, including [lfw]')
parser.add_argument('--test_data_path', default="", help='test data path')
parser.add_argument('--test_label_path', default="", help='test label path')
parser.add_argument('--target_face_path', default="", help='target attack face path')
parser.add_argument('--target_label_path', default="", help='target attack face label path')
parser.add_argument('--train_face_path', default="", help='training dataset path')
parser.add_argument('--train_dataset', default="", help='training dataset type, including [CelebA, CASIA, MS1M]')
parser.add_argument('--train_face_label_path', default="", help='training dataset label path')
parser.add_argument('--train_patch_path', default="", help='training dataset path')
parser.add_argument('--model', default="", help='number of classes')
parser.add_argument('--model_path', default="", help='number of classes')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--target_dataset', default='lfw', help='face data set')
parser.add_argument('--logfile', default='log.txt', help='log file to accord validation process')
parser.add_argument('--loss_acc_path', default='./loss_acc/train_loss/', help='save train loss as .p to draw pic')
parser.add_argument('--alpha', type=float, default=0.01, help='weight controls the attack loss')
parser.add_argument('--model_g_path',default='', help='save path of generator')
parser.add_argument('--model_d_path',default='', help='save path of discriminator')
parser.add_argument('--enable_new_loss',type=get_bool, default = False, help='whether enable tranditional GAN loss')
parser.add_argument('--patch_root',default='', help='save path of discriminator')
parser.add_argument('--patch_list',default='', help='save path of discriminator')
# parser.add_argument('--test_loss_acc_path',default='./loss_acc/train_acc/',help='save train acc as .p to draw pic')
parser.add_argument('--save_patch_path',default='', help='save path of discriminator')
parser.add_argument('--read_patch_path',default='', help='save path of discriminator')
parser.add_argument('--print_file',default='', help='save path of discriminator')
args = parser.parse_args()
def load_img(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    patch = Image.open(path).convert('RGB')
    patch = transform(patch)
    return patch
def save_img(img, path):
    img = img * 0.5 + 0.5
    img = np.array(img)
    img = img.transpose(1, 2, 0)
    img = img * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save(path)
from torch.optim import lr_scheduler
import copy
def train_op_optimize(model):
    totdev = 1
    npscal = NPSCalculator(args.print_file,64).cuda()
    totvacal = TotalVariation().cuda()
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count()," GPUs")
        totdev=torch.cuda.device_count()
        model = nn.DataParallel(model)
        npscal = nn.DataParallel(npscal)
        totvacal = nn.DataParallel(totvacal)
    elif torch.cuda.device_count() == 1:
        print("1 Gpu")
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #now the 3 trainset can be loaded in the same way
    trainset, testset = load_file(args.train_face_label_path, args.train_dataset)
    face_train_dataset = Train_Dataset(args.train_face_path, trainset, transform = transform)

    # lfw must be specially treated, as it is a verification dataset
    if args.target_dataset == 'lfw':
        lfw_dataset = LFW(root = args.target_face_path, file_list = args.target_label_path, transform=transform)
        target_face_loader = DataLoader(lfw_dataset, batch_size=args.face_batchsize, shuffle=False, drop_last=False)
        print('Total length of lfw:', target_face_loader.__len__())
    else:
        trainset, testset = load_file(args.train_face_label_path, args.train_dataset, test=True)
        face_target_dataset = Train_Dataset(args.target_face_path, trainset, transform = transform)
        target_face_loader = DataLoader(dataset = face_target_dataset, batch_size = 1, shuffle = False, drop_last = False)
        print('Total length of target face set: ', target_face_loader.__len__())
    
    face_train_loader = DataLoader(dataset = face_train_dataset, batch_size = args.face_batchsize, shuffle = True, drop_last = False)
    print('load train set')
    print('Total length of train face set: ', face_train_loader.__len__())
    #load adv patch from jpg image
    adv_patch_cpu = load_img(args.read_patch_path)
    adv_patch_cpu.requires_grad_(True)

    optimizer = torch.optim.Adam([adv_patch_cpu], lr=args.lr, weight_decay=1e-4,amsgrad=True)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[200,500,800], gamma=0.1)

    CE_loss = nn.CrossEntropyLoss()
    BCE_loss = nn.BCELoss()

    
    curr_lr = args.lr

    train_losses = []
    train_acc = []
    test_acc = []
    train_step = []
    test_step = []
    judge = 0
    print('start training')
    for epoch in range(args.epoch):
        scheduler.step()
        for step_target, (target_face, targetlabel) in enumerate(target_face_loader):
            target_face = target_face.repeat(args.face_batchsize,1,1,1)#stick patch to one face, or to different face but same identity
            target_face = Variable(target_face).cuda()
            #target_face_multi = Variable(target_face_multi).cuda()
            for step_face, (trainface, trainlabel) in enumerate(face_train_loader):
                x_face = Variable(trainface).cuda()
                adv_patch = adv_patch_cpu.cuda()

                adv_patch_multi = adv_patch_cpu.unsqueeze(0)
                adv_patch_multi = adv_patch_multi.repeat(args.face_batchsize,1,1,1)
                adv_patch_multi = adv_patch_multi.cuda()

                adv_face = stick_patch_on_face(copy.deepcopy(x_face), adv_patch_multi)
                adv_face = adv_face.cuda()
                nps = npscal(adv_patch)
                tv = totvacal(adv_patch)

                nps_loss = nps * 5
                tv_loss = tv * 10
                L_attack = predict(model, target_face, adv_face, 0.3)
                L_same = predict(model, x_face, adv_face, 0.3)

                L_attack = L_attack.cuda()
                L_same = L_same.cuda()

                attack_loss = 800 * (1 - L_attack) + 600 * L_same
                loss = attack_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                adv_patch_cpu.data.clamp_(-1,1)  

            if (step_target % (16//args.face_batchsize) == 0 ):
                print('epoch={}/{}'.format(epoch, args.epoch))
                print('saving image')
                save_img(adv_patch_cpu.detach(),args.save_patch_path)
                print('now l_attack: ', L_attack.detach().item())
                print('now l_same: ', L_same.detach().item())
                print('now nps loss: ', nps.item())
                print('now total variable: ', tv.item())
                    #acc = test_op(model,)
    # end for epoch

    #output_file.close()

def choose_model():
    # switch models
    #print(args.model)
    #now the best score is achieved on SE_ResNet_IR 50, which was proposed in ArcFace.
    if args.model == 'se_resnet_50':
        sub_model = SEResNet_IR(50, mode='se_ir')
        sub_model.load_state_dict(torch.load(args.model_path))
    elif args.model == 'resnet18':
        pass
    sub_model.cuda()
    return sub_model

if __name__ == "__main__":
    cnn = choose_model()
    train_op_optimize(cnn)
