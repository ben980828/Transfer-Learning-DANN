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
from sklearn.manifold import TSNE


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

        class_label = int(self.label_csv[np.where(
            self.label_csv == image_fn)[0].item()][1])
        domain_label = 1
        return image, class_label, domain_label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class Target_Image_2(Dataset):
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

        class_label = int(self.label_csv[np.where(
            self.label_csv == image_fn)[0].item()][1])
        domain_label = 0
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


class Feature_Extract(nn.Module):
    def __init__(self):
        super(Feature_Extract, self).__init__()
        self.feature_extract = nn.Sequential()
        self.feature_extract.add_module(
            'f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature_extract.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature_extract.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extract.add_module('f_relu1', nn.ReLU(True))
        self.feature_extract.add_module(
            'f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature_extract.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature_extract.add_module('f_drop1', nn.Dropout2d())
        self.feature_extract.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extract.add_module('f_relu2', nn.ReLU(True))

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature_extract = self.feature_extract(input_data)
        output = feature_extract.view(-1, 50*4*4)

        return output


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

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

    def forward(self, input_data):
        class_output = self.class_classifier(input_data)

        return class_output


class Domain(nn.Module):

    def __init__(self):
        super(Domain, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50*4*4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, epsilon):
        reverse_feature_extract = ReverseLayerF.apply(
            input_data, epsilon)  # gradient update dir change
        # gradient ascent for domain classification
        domain_output = self.domain_classifier(reverse_feature_extract)

        return domain_output


def main():
    # pyfile = sys.argv[0]
    # input_folder = sys.argv[1]
    # target_type = sys.argv[2]
    # output_file = sys.argv[3]

    input_folder = 'hw3-ben980828/hw3_data/digits/usps/test/'
    output_file = 'test_pred.csv'
    target_type = 'usps'
    source_type = 'mnistm'
    mode = 'acc'
            
    if target_type == 'usps':
        test_root = 'hw3-ben980828/hw3_data/digits/usps/test/'
        test_label_path = 'hw3-ben980828/hw3_data/digits/usps/test.csv'
    elif target_type == 'mnistm':
        test_root = 'hw3-ben980828/hw3_data/digits/mnistm/test/'
        test_label_path = 'hw3-ben980828/hw3_data/digits/mnistm/test.csv'
    elif target_type == 'svhn':
        test_root = 'hw3-ben980828/hw3_data/digits/svhn/test/'
        test_label_path = 'hw3-ben980828/hw3_data/digits/svhn/test.csv'

    test_list = os.listdir(test_root)

    test_label_file = pd.read_csv(test_label_path, sep=',',header=None)
    test_label_matrix = test_label_file.to_numpy()

    if source_type == 'usps':
        test_root_2 = 'hw3-ben980828/hw3_data/digits/usps/test/'
        test_label_path_2 = 'hw3-ben980828/hw3_data/digits/usps/test.csv'
    elif source_type == 'mnistm':
        test_root_2 = 'hw3-ben980828/hw3_data/digits/mnistm/test/'
        test_label_path_2 = 'hw3-ben980828/hw3_data/digits/mnistm/test.csv'
    elif source_type == 'svhn':
        test_root_2 = 'hw3-ben980828/hw3_data/digits/svhn/test/'
        test_label_path_2 = 'hw3-ben980828/hw3_data/digits/svhn/test.csv'

    test_list_2 = os.listdir(test_root_2)

    test_label_file_2 = pd.read_csv(test_label_path_2, sep=',',header=None)
    test_label_matrix_2 = test_label_file_2.to_numpy()


    test_set = Target_Image(fileroot=test_root,
                            image_root=test_list,
                            label_csv=test_label_matrix,
                            transform=transforms.Compose([
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])
                            )
    test_set_2 = Target_Image_2(fileroot=test_root_2,
                            image_root=test_list_2,
                            label_csv=test_label_matrix_2,
                            transform=transforms.Compose([
                                transforms.Resize((28, 28)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])
                            )
    target_set_size = int(len(test_set) * 0.8)
    valid_set_size = len(test_set) - target_set_size
    _, test_split = random_split(dataset= test_set, lengths=[target_set_size, valid_set_size])

    target_set_size = int(len(test_set_2) * 0.8)
    valid_set_size = len(test_set_2) - target_set_size
    _, test_split_2 = random_split(dataset= test_set_2, lengths=[target_set_size, valid_set_size])

    if mode == 'acc' or mode == 'tsne_class':
        test_loader = DataLoader(test_set, batch_size=1000,
                                shuffle=False, num_workers=1)
        test_loader_2 = DataLoader(test_set_2, batch_size=1000,
                                shuffle=False, num_workers=1)
    elif mode == 'tsne_domain':
        test_loader = DataLoader(test_split, batch_size=1000,
                                shuffle=False, num_workers=1)
        test_loader_2 = DataLoader(test_split_2, batch_size=1000,
                                shuffle=False, num_workers=1)
    correct = 0
    feature_extractor = Feature_Extract()
    classifier = Classifier()
    domain = Domain()
    if target_type == 'svhn':
        fe_state = torch.load('./dann_feature_extract_mnistm2svhn.pth')
        cs_state = torch.load('./dann_classifier_mnistm2svhn.pth')
        dom_state = torch.load('./dann_domain_mnistm2svhn.pth')
    elif target_type == 'usps':
        fe_state = torch.load('./dann_feature_extract_svhn2usps.pth')
        cs_state = torch.load('./dann_classifier_svhn2usps.pth')
        dom_state = torch.load('./dann_domain_svhn2usps.pth')
    elif target_type == 'mnistm':
        fe_state = torch.load('./dann_feature_extract_usps2mnistm.pth')
        cs_state = torch.load('./dann_classifier_usps2mnistm.pth')
        dom_state = torch.load('./dann_domain_usps2mnistm.pth')
    feature_extractor.load_state_dict(fe_state)
    classifier.load_state_dict(cs_state)
    domain.load_state_dict(dom_state)

    feature_extractor.to(device)
    classifier.to(device)
    domain.to(device)

    feature_extractor.eval()  # must remember
    classifier.eval()
    domain.eval()
    classifier_last2 = nn.Sequential(*list(classifier.children())[:-5])

    output = []
    output_y = []
    
    with torch.no_grad():  
        if mode == 'acc':
            input_list = []
            test_path = os.listdir(input_folder)
            for fn in test_path:
                input_list.append(fn)
            with open(output_file, 'w') as csvfile:
                csvfile.write('image_name,label\n')
                for i, filename in enumerate(input_list):
                    abs_path = os.path.join(input_folder, filename)
                    pil_image = Image.open(abs_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                    ])
                    image = transform(pil_image)
                    image_ch = torch.unsqueeze(image, 0)
                    model_input = image_ch.to(device)
                    feature_output = feature_extractor(model_input)
                    class_output = classifier(feature_output)
                    output_label = torch.argmax(class_output)
                    label = output_label.item()
                    csvfile.write('{},{}\n'.format(filename, label))

        elif mode == 'tsne_class':
            for i, (data, class_label, _) in enumerate(test_loader):
                data, class_label = data.cuda(), class_label.cuda()
                feature_output = feature_extractor(data)
                output4tsne = classifier_last2(feature_output)
                k = output4tsne.cpu().numpy()
                for i in class_label:
                    output_y.append(i.item())
                for i, data in enumerate(k):
                    output.append(k[i])
        elif mode == 'tsne_domain':
            for i, (data, _, domain_label) in enumerate(test_loader):
                p = float(i + 50 * len(test_loader)) / 50 / len(test_loader)
                eps = 2. / (1. + np.exp(-10 * p)) - 1
                data, domain_label = data.cuda(), domain_label.cuda()
                feature_output = feature_extractor(data)
                output4tsne = domain(feature_output, eps)
                k = output4tsne.cpu().numpy()
                for i in domain_label:
                    output_y.append(i.item())
                for i, data in enumerate(k):
                    output.append(k[i])
            for i, (data_2, _, domain_label_2) in enumerate(test_loader_2):
                data_2, domain_label_2 = data_2.cuda(), domain_label_2.cuda()
                feature_output_2 = feature_extractor(data_2)
                output4tsne_2 = domain(feature_output_2, eps)
                k_2 = output4tsne_2.cpu().numpy()
                for i in domain_label_2:
                    output_y.append(i.item())
                for i, data in enumerate(k_2):
                    output.append(k_2[i])
    if mode == 'tsne_class' or mode == 'tsne_domain':
        tsne = TSNE(n_components=2, n_iter=2000)

        tsne_results = tsne.fit_transform(output)
        x_min, x_max = tsne_results.min(0), tsne_results.max(0)
        X_norm = (tsne_results - x_min) / (x_max - x_min)  #Normalize
        plt.figure(figsize=(25, 25))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(output_y[i]), color=plt.cm.Set1(output_y[i]), 
                    fontdict={'weight': 'bold', 'size': 12})
        plt.xticks([])
        plt.yticks([])
        plt.savefig('dann_sr_{}_tr_{}_{}.png'.format(source_type, target_type, mode))

    

if __name__ == '__main__':
    main()
