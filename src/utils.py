
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

# TODO 2
# stick patch on face
# @return (tensor) faces with patch
def stick_patch_on_face(faceTensor, patchTensor):
    return None