from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os
import torch
# todo to be modified
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def default_loader(path, size, preprocess):
    img =  Image.open(path).convert('RGB')
    img = img.resize((size, size))
    if preprocess is not None:
        img = preprocess(img)
    else:
        img = transforms.ToTensor(img)
    return img


class local_Dataloader(Dataset):
    def __init__(self, img_path, label_path, img_size = 224, preprocess = None, loader = default_loader):

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print('No file path exists.')
            return

        fr = open(img_path, 'r')
        imgs = []
        for line in fr:
            line = line.strip('\n')
            line = line.rstrip()
            imgs.append(line)

        fr = open(label_path, 'r')
        labels = []
        for line in fr:
            line = line.strip('\n')
            line = line.rstrip()
            labels.append(line)

        self.image_list = imgs
        self.label_list = labels
        self.loader = loader
        self.img_size = img_size
        self.proprocess = preprocess

    def __getitem__(self, index):
        file_list = self.image_list[index]
        img = self.loader(file_list, self.img_size, self.preprocess)
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return len(self.image_list)



class local_Dataloader_no_label(Dataset):
    def __init__(self, img_path, img_size = 224, preprocess = None, loader = default_loader):

        if not os.path.exists(img_path):
            print('No file path exists.')
            return

        fr = open(img_path, 'r')
        imgs = []
        for line in fr:
            line = line.strip('\n')
            line = line.rstrip()
            imgs.append(line)

        self.image_list = imgs
        self.loader = loader
        self.img_size = img_size
        self.proprocess = preprocess

    def __getitem__(self, index):
        file_list = self.image_list[index]
        img = self.loader(file_list, self.img_size, self.preprocess)
        return img

    def __len__(self):
        return len(self.image_list)

#Create DataLoader for the following dataset:
'''
CASIA-WebFace
CelebA
MS-Celeb-1M
'''
#Because their ways of giving the image and label of the dataset are same,
#all being included in a txt file with line containing
#image file name on the left and the corresponding label on the right. See the examples:
'''
000001.jpg 1-txt file of CelebA
000045/1.jpg 0-txt file of both MS-1M and CASIA-WebFace
'''
#parameter:
# root: the root file folder of the images in dataset
# imgtext: the preprocessed txt file of 3 dataset

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, imgtext, transform=None):
        self.imgs = imgtext
        self.transform = transform
        self.root = root
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        try:
            img = Image.open(self.root + fn).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        except:
            print("failure %s" % fn)
            return None, None
        return img, label

    def __len__(self):
        return len(self.imgs)
    
def load_patch_file(root):
    trainset = []
    fh = open(root+'patch_one.txt', 'r')
    for line in fh:
        line = line.rstrip()
        words = line.split()
        trainset.append((words[0], int(words[1]) - 1))
    return trainset
def load_file(file_root, dataset, test=False):
    if dataset == 'CelebA':
        if(test):
            datafile = file_root + 'CelebA_test.txt'
        else:
            datafile = file_root + 'CelebA_list.txt'
        failfile = file_root + 'CelebA_fail.txt'
        trainset = []
        testset = []
        fh = open(datafile, 'r')
        failed = open(failfile, 'r')
        k = 0
        m = 0
        fail = []
        for lines in failed:
            lines = lines.rstrip()
            words = lines.split()
            fail.append(words[0])
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if (k <= 82):
                if (words[0] == fail[k]):
                    k = k + 1
                    if (k <= 50):
                        if (fail[k] == '101283.jpg'):
                            k = k + 1
                    continue
            if (words[0] == '101283.jpg'):
                continue
            elif (m % 10 == 9):
                testset.append((words[0], int(words[1]) - 1))
            else:
                trainset.append((words[0], int(words[1]) - 1))
            m = m + 1
        return trainset, testset
    if dataset == 'CASIA':
        if(test):
            datafile = file_root + 'CASIA_small_target.txt'
        else:
            datafile = file_root + 'CASIA_small_train.txt'
        failfile = file_root + 'CASIA_fail.txt'
        trainset = []
        testset = []
        fh = open(datafile, 'r')
        failed = open(failfile, 'r')
        k = 0
        m = 0
        fail = []
        for line in failed:
            line = line.rstrip()
            words = line.split()
            fail.append(words[1])
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if (k < 9099):
                if (words[0] == fail[k]):
                    k += 1
                    continue
            trainset.append((words[0], int(words[1])))
            m = m + 1
        return trainset, testset
    if dataset == 'MS1M':
        datafile = file_root + 'MS1M_list.txt'
        trainset = []
        testset = []
        fh = open(datafile, 'r')
        m = 0
        for line in fh:
            line = line.rstrip()
            words = line.split()
            if (words[0] == 'm.015t56/56-FaceId-0.jpg'):
                continue
            if (m % 500 == 4):
                testset.append((words[0], int(words[1])))
            else:
                trainset.append((words[0], int(words[1])))
            m = m + 1
        return trainset, testset
#test trainset building
if __name__ == '__main__':
    '''
    train_face_path='../dataset/'
    train_dataset='CASIA'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    trainset, testset = load_file(train_face_path, train_dataset, test=True)
    dataset = Train_Dataset('E:/CASIA/casia-112x112/', trainset, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    print(len(dataset))
    print(train_loader.__len__())
    tot=0
    for i, (images, labels) in enumerate(train_loader):
        print(images.size())
        print(labels.size())
        tot+=1
        print('----------')
        if(tot==1):
            break
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    import numpy as np
    set = load_patch_file('../')
    dataset = Train_Dataset('/userhome/dataset/patch/', set, transform=transform)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for i, (images, labels) in enumerate(train_loader):
        print(images.size())
        print(labels.size())
        images = np.array(images.squeeze(0))
        images  = images  .transpose(1, 2, 0)
        images  = images  * 255
        img = Image.fromarray(images.astype('uint8'))
        img.show()
        print('----------')
        break