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
from Model.SE_ResNet_IR import *
from utils import stick_patch_on_face,read_p_data
import torch.utils.data as Data
from generator import *
device = torch.device("cuda")
cudnn.benchmark = True

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
    img, img_ = transform(img), transform(img)
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    out1 = model(img)
    out2 = model(img_)
    ft = torch.cat((out1, out2), 1)[0].to('cpu')
    return ft
def extractDeepFeature_patch(G,patch, img, model, is_gray):
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
    img, img_ = transform(img), transform(img)
    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')
    patch = G(img).to('cuda')

    img = stick_patch_on_face(img, patch)
    img_ = stick_patch_on_face(img_,patch)
    out1 = model(img)
    out2 = model(img_)
    ft = torch.cat((out1, out2), 1)[0].to('cpu')
    return ft
def extractDeepFeature_predict(img, model, is_gray=False):
    img = img.to('cuda')#, img.to('cuda')
    out1 = model(img)
    #out2 = model(img_)
    #ft = torch.cat((out1, out2), 1)[0]
    return out1.cpu()


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


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def predict(model, target_face, train_face, best_threshold=0.3001, model_path=None):
    #print(target_face.size())
    #print(train_face.size())
    f1 = extractDeepFeature_predict(target_face, model)
    f2 = extractDeepFeature_predict(train_face, model)
    #print(f1.size())#[1024]
    #print(f2.size())
    for i in range(f1.size()[0]):
        if(i==0):
            distance = f1[i].dot(f2[i]) / (f1[i].norm() * f2[i].norm() + 1e-5)
        else:
            distance += f1[i].dot(f2[i]) / (f1[i].norm() * f2[i].norm() + 1e-5)
    #print(distance.size())
    distance = distance/f1.size()[0]
    return distance

def eval(model, model_path=None, is_gray=False):
    predicts = []
    #model.load_state_dict(torch.load(model_path))
    model.eval()
    root = '/media/dsg3/datasets/lfw/lfw_align/'
    with open('/media/dsg3/datasets/lfw/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]
    patch, _ = read_p_data('../dataset/doodle.p')
    patch_dataset = Data.TensorDataset(patch, _)
    patch_train_loader = Data.DataLoader(dataset=patch_dataset, batch_size=1, shuffle=False, drop_last=False)
    print('length of patch: ',patch_train_loader.__len__())
    G = StyleGenerator()
    G.load_state_dict(torch.load('/media/dsg3/FaceAttack/faceAttack_G.pkl'))
    G = G.cuda()
    tot = 10
    for i,(xx,_) in enumerate(patch_train_loader):
        if(i==tot):
            patch = xx
            break
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
            f2 = extractDeepFeature_patch(G,patch, img2, model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
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

    return np.mean(thd)


if __name__ == '__main__':
    #model=SEResNet_IR(50, mode='se_ir')
    #model_class = Resnet50FaceModel
    model=SEResNet_IR(50, mode='se_ir')
    result = eval(model.to('cuda'), model_path='/media/dsg3/xuyitao/Face/model/params_res50IR_cos_CA.pkl')
    #threshold = eval(model.to('cuda'), model_path='/media/dsg3/xuyitao/Face/model/params_res50IR_cos_MS_224.pkl')