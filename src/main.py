from __future__ import print_function
import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from src.SE_ResNet_IR import *
from src.lfw_DataLoader import *
from torch.optim.lr_scheduler import StepLR
import os
from src.myDataLoader import *

from src.utils import *
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
parser.add_argument('--model_path', default="", help='number of classes')
parser.add_argument('--batchnorm', type=get_bool, default=True, help='batch normalization')
parser.add_argument('--dropout', type=get_bool, default=True, help='dropout')
parser.add_argument('--target_dataset', default='lfw', help='face data set')
parser.add_argument('--logfile', default='log.txt', help='log file to accord validation process')
parser.add_argument('--loss_acc_path', default='./loss_acc/train_loss/', help='save train loss as .p to draw pic')
# parser.add_argument('--test_loss_acc_path',default='./loss_acc/train_acc/',help='save train acc as .p to draw pic')

args = parser.parse_args()

target_batchsize = args.face_batchsize * args.patch_batchsize

# todo 1
# model: substitute model for face recognition
# G, D: GAN
def train_op(model, G, D):

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
    trainset, testset = load_file(args.train_face_path, args.train_dataset)
    face_train_dataset = Train_Dataset(args.train_face_path + args.train_dataset, trainset, transform = transform)

    #original code:
    '''
    target_face_loader = DataLoader(local_Dataloader(img_path=args.target_face_path, label_path=args.target_face_label_path),
        batch_size=1, shuffle=True)

    face_train_loader = DataLoader(local_Dataloader(img_path= args.train_face_path,label_path= args.train_face_label_path),
                                   batch_size=args.face_batchsize, shuffle=True)
    '''

    # lfw must be specially treated, as it is a verification dataset
    if args.target_dataset == 'lfw':
        lfw_dataset = LFW(root = args.target_face_path, file_list = args.target_face_label_path, transform=transform)
        target_face_loader = DataLoader(lfw_dataset, batch_size=1, shuffle=False, drop_last=False)

    face_train_loader = DataLoader(dataset = face_train_dataset, batch_size = args.face_batchsize, shuffle = True, drop_last = True)

    '''
    patch_train_loader = DataLoader(local_Dataloader_no_label(img_path= args.train_patch_path),
                                    batch_size=args.patch_batchsize, shuffle=True)
    '''
    patch, _ = read_p_data(args.train_patch_path)
    patch_dataset = Data.TensorDataset(patch, _)
    patch_train_loader = Data.DataLoader(dataset=patch_dataset, batch_size=args.patch_batchsize, shuffle=False, drop_last=False)
    # original code:
    '''
    target_face_data = torch.Tensor(target_face).view(-1,3,32,32)[:target_batchsize].cuda() / 255.
    '''
    optimizer_g = torch.optim.Adam(G.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer_d = torch.optim.Adam(D.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

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

    for epoch in range(args.epoch):

        scheduler_g.step()
        scheduler_d.step()
        for step_target, (target_face, target_label) in target_face_loader:
            for step_face, (x_face, y_face) in face_train_loader:
                for step_patch, (x_patch, _) in patch_train_loader:

                    x_face = Variable(x_face).cuda()
                    x_patch = Variable(x_patch).cuda()

                    # feed target face to G to generate adv_patch
                    adv_patch = G(target_face)

                    # G loss
                    real_label = Variable(torch.ones(args.patch_batchsize)).cuda()
                    fake_label = Variable(torch.zeros(args.patch_batchsize)).cuda()

                    D_fake = D(adv_patch)
                    L_g = BCE_loss(D_fake, real_label)

                    # D loss
                    D_real = D(x_patch)
                    L_d = BCE_loss(D_real, real_label) + BCE_loss(D_fake, fake_label)

                    # stick adversarial patches on faces to generate adv face
                    adv_face = stick_patch_on_face(x_face, adv_patch)

                    # feed adv face to model
                    adv_logits = model(adv_face)

                    # attack loss
                    target_face_label = Variable(torch.full(target_batchsize, target_label[0][0])).cuda()
                    L_attack = CE_loss(adv_logits, target_face_label)

                    # overall loss
                    L_G = L_g + L_attack
                    L_D = L_d

                    # optimization
                    optimizer_g.zero_grad()
                    optimizer_d.zero_grad()

                    L_G.backward()
                    optimizer_g.step()

                    L_D.backward()
                    optimizer_d.step()


                # test acc for validation set
                if step_face % 10 == 0:
                    if args.enable_lat:
                        model.zero_reg()
                    #f.write('[Epoch={}/{}]: step={}/{},'.format(epoch, args.epoch, step, len(train_loader)))
                    print('epoch={}/{}'.format(epoch, args.epoch))
                    acc = test_op(model, output_file)


        # save model
        if epoch % 2 == 0:
            print('saving model...')
            torch.save(G.state_dict(), args.model_g_path + 'faceAttack_G.pkl')
            torch.save(D.state_dict(), args.model_d_path + 'faceAttack_D.pkl')


    # end for epoch

    output_file.close()


def test_op(model, G, f=None):
    '''
    test_target_face_loader = DataLoader(
        local_Dataloader(img_path=args.test_target_face_path, label_path=args.test_target_face_label_path),
        batch_size=1, shuffle=False)

    face_test_loader = DataLoader(
        local_Dataloader(img_path=args.train_face_path, label_path=args.test_face_label_path),
        batch_size=args.face_batchsize, shuffle=False)
    '''
    if args.test_dataset == 'lfw':
        lfw_dataset = LFW(root = args.test_target_face_path, file_list = args.test_target_face_label_path, transform=transform)
        test_target_face_loader = DataLoader(lfw_dataset, batch_size=1, shuffle=False, drop_last=False)
    else:
        print('not supported test dataset')
        test_target_face_loader = None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # now the 3 trainset can be loaded in the same way, in test_op we use the testset
    trainset, testset = load_file(args.train_face_path, args.train_dataset)
    face_test_dataset = Train_Dataset(args.train_face_path + args.train_dataset, testset, transform=transform)
    face_test_loader = DataLoader(dataset = face_test_dataset, batch_size=args.face_batchsize, shuffle=False, drop_last = False)
    # Test the model
    model.eval()
    G.eval()
    attack_succ = 0
    total = 0
    for step, (x, y) in test_target_face_loader:
        for iter, (test_face, _) in face_test_loader:

            x = Variable(x).cuda()
            test_face = Variable(test_face).cuda()
            #x_patch = Variable(x_patch).cuda()

            # feed target face to G to generate adv_patch
            adv_patch = G(x)
            adv_face = stick_patch_on_face(test_face, adv_patch)
            with torch.no_grad():
                h = model(adv_face)
            _, predicted = torch.max(h.data, 1)
            total += adv_face.size(0)
            attack_succ += (predicted == y).sum().item()
    acc = 100 * attack_succ / total
    print('Attack success rate on the test images: {:.2f} %'.format(acc))
    if f != None:
        f.write('Attack success rate on the test images: {:.2f} %'.format(acc))
        f.write('\n')
    # print('now is {}'.format(type(model)))
    model.train(True)
    return acc


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
    cnn = choose_model()

