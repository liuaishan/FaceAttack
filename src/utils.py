
import os
import numpy as np
import pickle
import torch
import torch.nn as nn

# read data without label from files
# @return (numpy array) data, len
def read_data_no_label(file_path):

    if not os.path.exists(file_path):
        return None, None, 0

    with open(file_path, 'rb') as fr:
        data_set = pickle.load(fr)
        size = len(data_set[0])
        list_data = []
        list_label = []
        # illegal data
        if not len(data_set[0]) == len(data_set[1]):
            return None, None, 0

        data = torch.unsqueeze(data_set[0], dim=1).type(torch.FloatTensor)[:size]

        data = np.asarray(data)
        return data, size


# read data from files
# @return (numpy array) data, len
def read_data(file_path):

    if not os.path.exists(file_path):
        return None, None, 0

    with open(file_path, 'rb') as fr:
        data_set = pickle.load(fr)
        size = len(data_set[0])
        list_data = []
        list_label = []
        # illegal data
        if not len(data_set[0]) == len(data_set[1]):
            return None, None, 0

        #data = data_set[0][:size] / 255.

        data = torch.unsqueeze(data_set[0], dim=1).type(torch.FloatTensor)[:size]
        label = data_set[1][:size]

        data = np.asarray(data)
        label = np.asarray(label)
        return data, label, size


def read_data_label(data_path, label_path):

    if not os.path.exists(data_path):
        return None, None, 0

    with open(data_path, 'rb') as fr:
        test_data = pickle.load(fr)
        size = len(test_data)
    with open(label_path, 'rb') as fr:
        test_label = pickle.load(fr)
    return test_data, test_label, size

def read_p_data(data_path):
    with open(data_path, 'rb') as fr:
        data = pickle.load(fr)
    return data, torch.ones(len(data))


# TODO 2
# stick patch on face
# @return (tensor) faces with patch
def stick_patch_on_face_old(faceTensor, patchTensor):
   
    # position of patch
    x=140
    y=160
    
    # get batch size of face dataset and patch dataset
    face_bsize = len(faceTensor)
    patch_bsize = len(patchTensor)
    
    #get size of patch data
    size1 = len(patchTensor[0][0])
    size2 = len(patchTensor[0][0][0])

    # stick every m patch data on k face data
    for k in range(face_bsize):
        face = faceTensor[k]
        for m in range(patch_bsize):
            patch = patchTensor[m]
            for i in range(size1):   
                 for j in range(size2):
                     face[0][x+i][y+j] = patch[0][i][j]
                     face[1][x+i][y+j] = patch[1][i][j] 
                     face[2][x+i][y+j] = patch[2][i][j]
            if k==0 and m==0:
                new = face.unsqueeze(0)
                combineTensor = new
            else:  
                new = face.unsqueeze(0)
                combineTensor = torch.cat([combineTensor,new],0)
                
    return combineTensor
def stick_patch_on_face(faceTensor, patchTensor,x,y):
   
    # position of patch
    #x=140
    #y=160
    
    # get batch size of face dataset and patch dataset
    face_bsize = len(faceTensor)
    patch_bsize = len(patchTensor)
    
    all_bsize = face_bsize# * patch_bsize
    combineTensor = torch.zeros((all_bsize, faceTensor.size()[1], faceTensor.size()[2], faceTensor.size()[3]))
    k = 0
    for i in range(face_bsize):
        #for j in range(patch_bsize):
        faceTensor[i][:, x[i] : x[i]+patchTensor.size()[2], y[i]:y[i] + patchTensor.size()[3]] = patchTensor[i][:, :, :] 
        combineTensor[k] = faceTensor[i]
        k = k + 1
    
    return combineTensor
def get_face_part(faceTensor,patchTensor):
    x=140
    y=160
    
    # get batch size of face dataset and patch dataset
    face_bsize = len(faceTensor)
    patch_bsize = len(patchTensor)
    
    all_bsize = face_bsize# * patch_bsize
    combineTensor = torch.zeros((all_bsize, patchTensor.size()[1], patchTensor.size()[2], patchTensor.size()[3]))
    k = 0
    for i in range(face_bsize):
        combineTensor[k] = faceTensor[i][:, x : x+patchTensor.size()[2], y:y + patchTensor.size()[3]]
        k = k + 1
    
    return combineTensor