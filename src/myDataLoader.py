from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os

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
    img =  Image.open(path)
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