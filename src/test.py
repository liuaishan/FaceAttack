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

mpl.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('bmh')

from torch.autograd import Variable


def get_bool(string):
    if (string == 'False'):
        return False
    else:
        return True
def train_op_onlfw(model, G, D, nowbest_threshold):
    G=G.cuda()
    D=D.cuda()
    totdev = 1
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count()," GPUs")
        totdev=torch.cuda.device_count()
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("1 Gpu")
    model.eval()
    output_file = open(args.logfile, 'w')
    # load training data and test set
    #face_train, face_train_label, _ = read_data(args.train_face_path)
    #patch_train, _ = read_data_no_label(args.train_patch_path)
    #face_test, face_test_label, _ = read_data(args.test_face_path)
    #target_face, target_label, _ = read_data(args.target_face_path)

    # todo 1
    # preprocessing for different face dataset
    # including: normalization, transformation, etc.
    # added by xyt:
    # Training dataset must be built by the corresponding txt file, Already added such process in myDataLoader
    # As a result, the method local_Dataloader is abandoned
    # LFW dataset is for face verification, the way of traversing it is slightly strange.
    # todo 1.1
    # add more test dataset, especially for face recognition. MegaFace is the best one. However, using MegaFace
    # is a little bit tough. I did not realize the dataloader.
    '''
    original code:
    if args.dataset == 'lfw':
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor()])
        target_face_data = torch.Tensor(target_face).view(-1, 3, 512, 512)[:target_batchsize].cuda() / 255.
        target_face_label = torch.Tensor(target_label)[:args.target_batchsize].cuda()
    '''
    #To all the dataset, we do not distinguish the transform process so that we can make the train and test process unified
    # which includes resize, totensor and normalize(-1,1)
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #now the 3 trainset can be loaded in the same way
    trainset, testset = load_file(args.train_face_label_path, args.train_dataset)
    face_train_dataset = Train_Dataset(args.train_face_path, trainset, transform = transform)

    #original code:
    '''
    target_face_loader = DataLoader(local_Dataloader(img_path=args.target_face_path, label_path=args.target_face_label_path),
        batch_size=1, shuffle=True)

    face_train_loader = DataLoader(local_Dataloader(img_path= args.train_face_path,label_path= args.train_face_label_path),
                                   batch_size=args.face_batchsize, shuffle=True)
    '''

    # lfw must be specially treated, as it is a verification dataset
    if args.target_dataset == 'lfw':
        lfw_dataset = LFW(root = args.target_face_path, file_list = args.target_label_path, transform=transform)
        target_face_loader = DataLoader(lfw_dataset, batch_size=args.face_batchsize, shuffle=False, drop_last=False)
        print('Total lenght of lfw:', target_face_loader.__len__())
    face_train_loader = DataLoader(dataset = face_train_dataset, batch_size = args.face_batchsize, shuffle = True, drop_last = True)
    print('load train set')
    '''
    patch_train_loader = DataLoader(local_Dataloader_no_label(img_path= args.train_patch_path),
                                    batch_size=args.patch_batchsize, shuffle=True)
    '''
    patch, _ = read_p_data(args.train_patch_path)
    patch_dataset = Data.TensorDataset(patch, _)
    patch_train_loader = Data.DataLoader(dataset=patch_dataset, batch_size=args.patch_batchsize, shuffle=False, drop_last=False)
    print('load patch set')
    # original code:
    '''
    target_face_data = torch.Tensor(target_face).view(-1,3,32,32)[:target_batchsize].cuda() / 255.
    '''
    optimizer_g = torch.optim.Adam(G.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=args.lr, weight_decay=5e-4)

    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90

    scheduler_g = StepLR(optimizer_g, step_size=30, gamma=0.1)
    scheduler_d = StepLR(optimizer_d, step_size=30, gamma=0.1)
    CE_loss = nn.CrossEntropyLoss()
    BCE_loss = nn.BCELoss()

    curr_lr = args.lr

    train_losses = []
    train_acc = []
    test_acc = []
    train_step = []
    test_step = []
    print('start training G and D')
    for epoch in range(args.epoch):

        scheduler_g.step()
        scheduler_d.step()
        for step_target, (target_face) in enumerate(target_face_loader):
            for step_patch, (x_patch,_) in enumerate(patch_train_loader):
                x_face = Variable(target_face[0]).cuda()
                x_patch = Variable(x_patch).cuda()
                target_face[1] = Variable(target_face[1]).cuda()
                # feed target face to G to generate adv_patch
                adv_patch = G(target_face[1])
                #print(adv_patch.size())
                # G loss
                real_label = Variable(torch.ones(args.patch_batchsize)).cuda()
                fake_label = Variable(torch.zeros(args.patch_batchsize)).cuda()
                #real_label=real_label.unsqueeze(0)
                #fake_label=fake_label.unsqueeze(0)
                D_fake = D(adv_patch)
                #D_fake = Variable(torch.Tensor([0.2])).cuda()
                
                #D_fake = D_fake
                #print(D_fake.size())#1,1
                #print(real_label.size())#1
                #print(D_fake)
                #print(real_label,fake_label)
                L_g = BCE_loss(D_fake, real_label)#

                # D loss
                D_real = D(x_patch)
                #print(D_real)
                L_d = BCE_loss(D_real, real_label) + BCE_loss(D_fake, fake_label)
                
                # stick adversarial patches on faces to generate adv face
                adv_face = stick_patch_on_face(x_face, adv_patch)

                # feed adv face to model
                adv_feature = model(adv_face)

                # attack loss
                #target_face_label = Variable(torch.full(target_batchsize, target_label[0][0])).cuda()
                #L_attack = CE_loss(adv_logits, target_face_label)
                sameflag, L_attack = predict(model, target_face[0], target_face[1], best_threshold= nowbest_threshold)
                #print(sameflag)
                L_attack = L_attack.cuda()
                sameflag = torch.Tensor([sameflag]).cuda()
                
                #print(L_attack.size())
                #print(L_attack)
                # overall loss
                L_G = L_g + args.alpha * (sameflag * L_attack + (1 - sameflag) * (1 - L_attack))
                L_D = L_d

                # optimization
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                L_G.backward(retain_graph=True)
                optimizer_g.step()

                L_D.backward(retain_graph=True)
                optimizer_d.step()
                if(step_patch % 16 == 0):
                    print('now step in target face: ', step_target)
                    Loss_G = '%.2f' % L_G.item()
                    Loss_D = '%.2f' % L_D.item()
                    print('now G loss: ',Loss_G)
                    print('now D loss: ',Loss_D)
                    output_file.write('now step in target '+str(step_target)+'\n')
                    output_file.write('now G loss: '+str(Loss_G)+'\n')
                    output_file.write('now D loss: '+str(Loss_D)+'\n')
                if(step_patch >= 32):
                    break

            # test acc for validation set
            if step_target % 50 == 0:
                #if args.enable_lat:
                   # model.zero_reg()
                #f.write('[Epoch={}/{}]: step={}/{},'.format(epoch, args.epoch, step, len(train_loader)))
                print('epoch={}/{}'.format(epoch, args.epoch))
                print('saving model...')
                if(totdev==1):
                    torch.save(G.state_dict(), args.model_g_path + 'faceAttack_G.pkl')
                    torch.save(D.state_dict(), args.model_d_path + 'faceAttack_D.pkl')
                else:
                    torch.save(G.module.state_dict(), args.model_g_path + 'faceAttack_G.pkl')
                    torch.save(D.module.state_dict(), args.model_d_path + 'faceAttack_D.pkl')
                #acc = test_op(model,)


        # save model
        if epoch % 2 == 0:
            print('saving model...')
            torch.save(G.state_dict(), args.model_g_path + 'faceAttack_G.pkl')
            torch.save(D.state_dict(), args.model_d_path + 'faceAttack_D.pkl')


    # end for epoch

    output_file.close()

def test_op(G, f=None):
    from PIL import Image
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    lfw_dataset = LFW(root = '/media/dsg3/datasets/lfw/lfw_align/', file_list = '/media/dsg3/datasets/lfw/pairs.txt', transform=transform)
    target_face_loader = DataLoader(lfw_dataset, batch_size=1, shuffle=False, drop_last=False)
    print('Total lenght of lfw:', target_face_loader.__len__())
    for i,face in enumerate(target_face_loader):
        patch = G(face[1])
        patch = np.array(patch.detach().squeeze(0).numpy())
        patch = patch.transpose(1,2,0)
        patch *= 255
        img = Image.fromarray(patch.astype('uint8'))
        img.save('./'+str(i)+'.jpg')
        if(i==4):
            break

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

def load_model(g_path=None, d_path=None):
    G = StyleGenerator()
    D = StyleDiscriminator()
    if os.path.exists(g_path) == False:
        print('Load Generator failed')
    else:
        print('Successfully load G')
        G.load_state_dict(torch.load(g_path))
    if os.path.exists(d_path) == False:
        print('Load Discriminator failed')
    else:
        print('Successfully load D')
        D.load_state_dict(torch.load(d_path))
    return G, D


if __name__ == "__main__":
    '''
    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path)


    if os.path.exists(args.loss_acc_path) == False:
        os.makedirs(args.loss_acc_path)

    cnn = choose_model()

    if os.path.exists(args.model_path):
        cnn.load_state_dict(torch.load(args.model_path))
        print('load substitute model.')
    else:
        print("load substitute failed.")

    if args.test_flag:
        test_op(cnn)
    else:
        train_op(cnn)
    '''
    #cnn = choose_model()
    #nowbest_threshold = eval(cnn, model_path='')
    #nowbest_threshold=0.29
    #print('Now the best threshold on lfw is: ',nowbest_threshold)
    G, D = load_model('/media/dsg3/FaceAttack/faceAttack_G.pkl','/media/dsg3/FaceAttack/faceAttack_D.pkl')
    test_op(G)
