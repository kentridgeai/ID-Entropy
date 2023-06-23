import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math
# from utils_o import *
import torch.nn.functional
import numpy as np
from copy import copy



from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling


# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import torch.optim as optim
# from scipy import signal
# import torchvision
# from scipy.ndimage.filters import gaussian_filter
# import torchvision.transforms as transforms
# import sys
# from copy import copy
# import pickle
# from torch.nn.modules.utils import _pair, _quadruple
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# # The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# from torchvision.transforms import *
# sys.path.append('../../Libraries')
from torch.optim.lr_scheduler import StepLR
# from invariance_estimation import *

import os


class ConvolutionLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(ConvolutionLayer, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (int(padding), int(padding))
        self.conv = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)




class Net_vanilla_cnn_mnist(nn.Module):
    def __init__(self):
        super(Net_vanilla_cnn_mnist, self).__init__()

        kernel_sizes = [3, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [25, 45, 60,200]

        # network layers
        self.conv1 = ConvolutionLayer(1, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1, padding=pads[1])
        self.conv3 = ConvolutionLayer(layers[1], layers[2], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])
        # self.conv4 = ConvolutionLayer(layers[2], layers[3], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(layers[1])
        self.pool3 = nn.MaxPool2d(kernel_size=(7, 7))
        self.bn3 = nn.BatchNorm2d(layers[2])
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2))
        self.bn4 = nn.BatchNorm2d(layers[3])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[2], 50, 1)
        self.fc1bn = nn.BatchNorm2d(50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.2)
        self.fc2 = nn.Conv2d(50, 10, 1)

    def forward(self, x):
        # print(x.shape)
        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        feats0 = copy(xm.detach())
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        feats1 = copy(xm.detach())
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.bn2(x)
        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        feats2 = copy(xm.detach())
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)
        xm = self.bn3(x)

        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)

        # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)
        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        feats3 = copy(xm.detach())
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        # xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,[feats0,feats1,feats2,feats3]


class Net_vanilla_cnn_stl10(nn.Module):
    def __init__(self):
        super(Net_vanilla_cnn_stl10, self).__init__()

        kernel_sizes = [3, 3, 3]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        layers = [30, 60, 80]

        # network layers
        self.conv1 = ConvolutionLayer(1, layers[0], [kernel_sizes[0], kernel_sizes[0]], stride=1, padding=pads[0])
        self.conv2 = ConvolutionLayer(layers[0], layers[1], [kernel_sizes[1], kernel_sizes[1]], stride=1, padding=pads[1])
        self.conv3 = ConvolutionLayer(layers[1], layers[2], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])
        # self.conv4 = ConvolutionLayer(layers[2], layers[3], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])
        # self.conv5 = ConvolutionLayer(layers[3], layers[4], [kernel_sizes[2], kernel_sizes[2]], stride=1, padding=pads[2])

        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.bn2 = nn.BatchNorm2d(layers[1])
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4))
        self.bn3 = nn.BatchNorm2d(layers[2])
        # self.pool4 = nn.MaxPool2d(kernel_size=(2,2))
        # self.bn4 = nn.BatchNorm2d(layers[3])
        # self.pool5 = nn.MaxPool2d(2)
        # self.bn5 = nn.BatchNorm2d(layers[4])
        self.fc1 = nn.Conv2d(layers[2]*4, 100, 1)
        self.fc1bn = nn.BatchNorm2d(100)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(100, 10, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)
        xm = self.bn3(x)
        # print(x.shape)
        # x = self.conv4(x)
        # # print(x.shape)
        # x = self.pool4(x)
        # # print(x.shape)
        # xm = self.bn4(x)
        # x = self.conv5(x)
        # x = self.pool5(x)
        # xm = self.bn5(x)
        # xm = self.bn3_mag(xm)
        # print(xm.shape)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm




class MLP_mnist(nn.Module):
    def __init__(self):
        super(MLP_mnist, self).__init__()

        layers = [100,100]
        self.fc1 = nn.Conv2d(28*28, layers[0], 1)
        self.fc1bn = nn.BatchNorm2d(layers[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(layers[0], layers[1], 1)
        self.fc3 = nn.Conv2d(layers[1], 10, 1)
        self.fc2bn = nn.BatchNorm2d(layers[1])

    def forward(self, x):
        # print(x.shape)
        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])

        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        # xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = self.relu(self.fc2bn(xm))
        feats = copy(xm.detach())
        xm = self.fc3(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm,feats

class MLP_stl10(nn.Module):
    def __init__(self):
        super(MLP_stl10, self).__init__()

        layers = [10, 100]
        self.fc1 = nn.Conv2d(96 * 96, layers[0], 1)
        self.fc1bn = nn.BatchNorm2d(layers[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.7)
        self.fc2 = nn.Conv2d(layers[0], 10, 1)

    def forward(self, x):
        # print(x.shape)
        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])

        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])

        return xm



class GCNN_mnist(nn.Module):
    def __init__(self):
        super(GCNN_mnist, self).__init__()
        self.conv1 = P4ConvZ2(1, 10, kernel_size=3,padding=1)
        self.conv2 = P4ConvP4(10, 15, kernel_size=3,padding=1)
        self.conv3 = P4ConvP4(15, 19, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(76, 10)
        self.fc1bn = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = plane_group_spatial_max_pooling(x, 3, 3)
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 3, 3)
        x = F.relu(self.conv3(x))
        x = plane_group_spatial_max_pooling(x, 3, 3)
        # [x,temp] = torch.max(x,2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1bn(self.fc1(x)))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x





from torchvision import models
class resnet18_bertTuned(nn.Module):
    def __init__(self, num_classes):
        super(resnet18_bertTuned, self).__init__()
        self.resnet_feats = models.resnet18(pretrained=True)
        num_ftrs = self.resnet_feats.fc.in_features

        self.resnet_feats.fc = nn.Linear(num_ftrs, 768)
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, x):
        feats = nn.ReLU()(self.resnet_feats(x))
        logits = self.fc1(feats)

        return [feats,logits]



class GCNN_stl10(nn.Module):
    def __init__(self):
        super(GCNN_stl10, self).__init__()
        self.conv1 = P4ConvZ2(1, 20, kernel_size=3,padding=1)
        self.conv2 = P4ConvP4(20, 30, kernel_size=3,padding=1)
        self.conv3 = P4ConvP4(30, 60, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(60*4, 23)
        self.fc2 = nn.Linear(23, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = plane_group_spatial_max_pooling(x, 3, 3)
        x = F.relu(self.conv2(x))
        x = plane_group_spatial_max_pooling(x, 4, 4)
        x = F.relu(self.conv3(x))
        x = plane_group_spatial_max_pooling(x, 4, 4)
        # [x,temp] = torch.max(x,2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class ConvAe(nn.Module):
    def __init__(self):
        super(ConvAe, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation

        xm = x.view([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], 1, 1])
        feats = copy(xm.detach())
        # print(feats.shape)

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))

        return x,[feats]



class ConvAe_f(nn.Module):
    def __init__(self):
        super(ConvAe_f, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # print(encoded.shape)
        xm = encoded.view([encoded.shape[0], encoded.shape[1] * encoded.shape[2] * encoded.shape[3], 1, 1])
        feats = copy(xm.detach())
        decoded = self.decoder(encoded)
        return decoded, [feats]

