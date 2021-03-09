import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Function
import imageio
import scipy.misc
import argparse
import glob
import os
import sys
import csv
import pandas as pd
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

class Target_Image(Dataset):
    def __init__(self, fileroot, image_root, label_csv, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.fileroot = fileroot
        self.image_root = image_root
        self.transform = transform
        self.label_csv = label_csv
        
        self.len = len(self.image_root) 

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.image_root[index]
        image_path = os.path.join(self.fileroot, image_fn)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image) 

        class_label = int(self.label_csv[np.where(self.label_csv==image_fn)[0].item()][1])
        domain_label = 1
        return image, class_label, domain_label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, epsilon):
        ctx.epsilon = epsilon

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.epsilon

        return output, None

class DANN(nn.Module):

    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extract = nn.Sequential()
        self.feature_extract.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature_extract.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature_extract.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extract.add_module('f_relu1', nn.ReLU(True))
        self.feature_extract.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature_extract.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature_extract.add_module('f_drop1', nn.Dropout2d())
        self.feature_extract.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extract.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50*4*4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50*4*4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, epsilon):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature_extract = self.feature_extract(input_data)
        feature_extract = feature_extract.view(-1, 50*4*4)
        reverse_feature_extract = ReverseLayerF.apply(feature_extract, epsilon)#gradient update dir change
        class_output = self.class_classifier(feature_extract)
        domain_output = self.domain_classifier(reverse_feature_extract)#gradient ascent for domain classification

        return class_output, domain_output

def main():
    
    target_type = 'usps'
    if target_type == 'usps':
        target_root = 'hw3-ben980828/hw3_data/digits/usps/test/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/usps/test.csv'
    elif target_type == 'mnistm':
        target_root = 'hw3-ben980828/hw3_data/digits/mnistm/test/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/mnistm/test.csv'
    elif target_type == 'svhn':
        target_root = 'hw3-ben980828/hw3_data/digits/svhn/test/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/svhn/test.csv'

    target_list = os.listdir(target_root)

    target_label_file = pd.read_csv(target_label_path, sep=',',header=None)
    target_label_matrix = target_label_file.to_numpy()

    if target_type == 'usps':
        target_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.1307,], [0.3081,])
            ])
    else:
        target_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    target_set = Target_Image(fileroot=target_root, 
        image_root=target_list, 
        label_csv=target_label_matrix, 
        transform=target_transform
        )

    test_loader = DataLoader(target_set, batch_size=1000, shuffle=False, num_workers=1)

    correct = 0
    model = DANN()
    state = torch.load('dann_nosr_tr_usps_ep43_acc98.90335846470185.pth')
    model.load_state_dict(state)
    # print(model)
    model.to(device)
    # print(model)
    model.eval()#must remember
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (data, class_label, domain_label) in enumerate(test_loader):
            p = float(i + 1 * len(test_loader)) / 10 / len(test_loader)
            eps = 2. / (1. + np.exp(-10 * p)) - 1
            data, class_label, domain_label = data.cuda(), class_label.cuda(), domain_label.cuda()
            class_output, _ = model(data, eps)
            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(class_label.view_as(pred)).sum().item()
        acc = 100. * correct / len(test_loader.dataset)
        print('\nTest Set : Acc = {}\n'.format(acc))


if __name__ == '__main__':
    main()