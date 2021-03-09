import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

class Source_Image(Dataset):
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
        domain_label = 0
        return image, class_label, domain_label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

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
        self.feature_extract.add_module('f_drop1', nn.Dropout2d(0.5))
        self.feature_extract.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extract.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50*4*4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout(0.5))
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
    source_type = 'svhn'
    target_type = 'usps'
    if source_type == 'usps':
        source_root = 'hw3-ben980828/hw3_data/digits/usps/train/'
        source_label_path = 'hw3-ben980828/hw3_data/digits/usps/train.csv'
    elif source_type == 'mnistm':
        source_root = 'hw3-ben980828/hw3_data/digits/mnistm/train/'
        source_label_path = 'hw3-ben980828/hw3_data/digits/mnistm/train.csv'
    elif source_type == 'svhn':
        source_root = 'hw3-ben980828/hw3_data/digits/svhn/train/'
        source_label_path = 'hw3-ben980828/hw3_data/digits/svhn/train.csv'

    source_list = os.listdir(source_root)

    source_label_file = pd.read_csv(source_label_path, sep=',',header=None)
    source_label_matrix = source_label_file.to_numpy()
        
    if target_type == 'usps':
        target_root = 'hw3-ben980828/hw3_data/digits/usps/train/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/usps/train.csv'
    elif target_type == 'mnistm':
        target_root = 'hw3-ben980828/hw3_data/digits/mnistm/train/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/mnistm/train.csv'
    elif target_type == 'svhn':
        target_root = 'hw3-ben980828/hw3_data/digits/svhn/train/'
        target_label_path = 'hw3-ben980828/hw3_data/digits/svhn/train.csv'

    target_list = os.listdir(target_root)

    target_label_file = pd.read_csv(target_label_path, sep=',',header=None)
    target_label_matrix = target_label_file.to_numpy()
        
    if source_type == 'usps':
        source_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.1307,], [0.3081,])
            ])
    else:
        source_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

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

    source_set = Source_Image(fileroot=source_root, 
        image_root=source_list,
        label_csv=source_label_matrix,  
        transform=source_transform
        )

    target_set = Target_Image(fileroot=target_root, 
        image_root=target_list, 
        label_csv=target_label_matrix, 
        transform=target_transform
        )

    target_set_size = int(len(target_set) * 0.8)
    valid_set_size = len(target_set) - target_set_size
    _, valid_split = random_split(dataset= target_set, lengths=[target_set_size, valid_set_size])

    source_loader = DataLoader(source_set, batch_size=128, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_split, batch_size=700, shuffle=False, num_workers=1)


    dann = DANN()
    dann = dann.cuda()

    print(dann)


    lr = 1e-4
    loss_class = nn.NLLLoss()
    loss_domain = nn.NLLLoss()
    optimizer = optim.Adam(dann.parameters(), lr=lr, betas=(0.5, 0.999))

    epoch = 50
    iteration = 0
    max_acc = 0.
    class_loss_list = []
    domain_loss_list = []
    iter_list = []
    # training
    for ep in range(1, epoch+1):
        dann.train()
        print('Current training epoch : ', ep)

        for i, (data, class_label, domain_label) in enumerate(source_loader):
            p = float(i + ep * len(source_loader)) / epoch / len(source_loader)
            eps = 2. / (1. + np.exp(-10 * p)) - 1
            dann.zero_grad()
            data, class_label, domain_label = data.cuda(), class_label.cuda(), domain_label.cuda()
            class_output, domain_output = dann(data, eps)
            class_loss = loss_class(class_output, class_label)
            domain_loss = loss_domain(domain_output, domain_label)
            total_loss = class_loss + domain_loss

            total_loss.backward()
            optimizer.step()
            
            iteration += 1
            if iteration%50 == 0:
                iter_list.append(iteration)
                class_loss_list.append(class_loss.item())
                domain_loss_list.append(domain_loss.item())
            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], class_loss: %f, domain_loss: %f' \
                % (ep, i + 1, len(source_loader), class_loss.data.cpu().numpy(),
                    domain_loss.data.cpu().numpy()))
            sys.stdout.flush()

        dann.eval()
        correct = 0
        acc = 0
        with torch.no_grad(): 
            for i, (data, class_label, domain_label) in enumerate(valid_loader):
                p = float(i + ep * len(valid_loader)) / epoch / len(valid_loader)
                eps = 2. / (1. + np.exp(-10 * p)) - 1
                data, class_label, domain_label = data.cuda(), class_label.cuda(), domain_label.cuda()
                class_output, _ = dann(data, eps)
                pred = class_output.max(1, keepdim=True)[1] 
                correct += pred.eq(class_label.view_as(pred)).sum().item()
            acc = 100. * correct / len(valid_loader.dataset)
            print('\nValidation Set : Acc = {}\n'.format(acc))
        
        if acc > max_acc:
            print('Performance improved : ({:.3f} --> {:.3f}). Save model ==> '.format(max_acc, acc))
            max_acc = acc
            torch.save(dann.state_dict(), 'dann_sr_{}_tr_{}_ep{}_acc{}.pth'.format(source_type, target_type, ep, acc))


        #lr_decay.step(mean_loss)



    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(iter_list, class_loss_list)
    ax1.set_title('Classification Loss')
    ax1.set(xlabel="iteration", ylabel="Loss Value")


    ax2.plot(iter_list, domain_loss_list)
    ax2.set_title('Domain Loss')
    ax2.set(xlabel="iteration", ylabel="Loss Value")

    plt.savefig('Loss_Curve_DANN_Source.png')

if __name__ == '__main__':
    main()
