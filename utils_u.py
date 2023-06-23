import os
import pickle
import math
import numpy as np
import scipy.misc
# import cv2

import torch
import torchvision
from torch.utils import data
from torch.autograd import Variable

import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage import color


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))

        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding

        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        #  trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)

        if trim_top<0:
            out = img
            print(zoom_factor)

        else:
            out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img

    return out
#
#
# def preprocess_image(cv2im, resize_im=True):
#     """
#         Processes image for CNNs
#
#     Args:
#         PIL_img (PIL_img): Image to process
#         resize_im (bool): Resize to 224 or not
#     returns:
#         im_as_var (Pytorch variable): Variable that contains processed float tensor
#     """
#     # mean and std list for channels (Imagenet)
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     # Resize image
#     if resize_im:
#         cv2im = cv2.resize(cv2im, (224, 224))
#     im_as_arr = np.float32(cv2im)
#     im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
#     im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
#     # Normalize the channels
#     for channel, _ in enumerate(im_as_arr):
#         im_as_arr[channel] /= 255
#         im_as_arr[channel] -= mean[channel]
#         im_as_arr[channel] /= std[channel]
#     # Convert to float tensor
#     im_as_ten = torch.from_numpy(im_as_arr).float().cuda()
#     # Add one more channel to the beginning. Tensor shape = 1,3,224,224
#     im_as_ten.unsqueeze_(0)
#     # Convert to Pytorch variable
#     im_as_var = Variable(im_as_ten, requires_grad=True)
#     return im_as_var
#
#
# class FeatureVisualization():
#     def __init__(self, img_path, model, selected_layer):
#         self.img_path = img_path
#         self.selected_layer = selected_layer
#         self.pretrained_model = model
#
#     def process_image(self):
#         img = cv2.imread(self.img_path)
#         img = preprocess_image(img, resize_im=False)
#         return img
#
#     def get_feature(self):
#         # input = Variable(torch.randn(1, 3, 224, 224))
#         input = self.process_image()
#         print(input.shape)
#
#         x = input
#         for index, layer in enumerate(self.pretrained_model.modules()):
#             if layer._get_name() != 'ScaleConv_steering':
#             # if layer._get_name() != 'ConvolutionLayer':
#                 continue
#             print('index: ', index, ',', ' layer:', layer._get_name())
#             x = layer(x)
#             if index == self.selected_layer:
#                 return x
#
#     def get_single_feature(self):
#         features = self.get_feature()
#         print(features.shape)
#
#         feature = features[:, 0, :, :]
#         print(feature.shape)
#
#         feature = feature.view(feature.shape[1], feature.shape[2])
#         print(feature.shape)
#
#         return feature
#
#     def get_kernel_map(self, features):
#         feature_map = []
#         img_num, kernel_num, kernel_rows, kerner_cols = features.shape
#         map_size = int(math.sqrt(kernel_num)) + 1
#
#         for image_feature in features:
#             print(image_feature.shape)
#             for idx, feature in enumerate(image_feature):
#                 for _ in range(map_size):
#                     for _ in range(map_size):
#                         ax = plt.subplot(map_size, map_size, idx+1)
#                         ax.set_xticks([])
#                         ax.set_yticks([])
#                         pics = plt.imshow(feature.cpu().detach().numpy(), cmap='gray')
#             plt.savefig('./example/kernel_map_steerable.png')
#         plt.show()
#
#     def get_and_save_all_feature(self):
#         kernel_map = []
#         features = self.get_feature()
#         print(features.shape)
#         self.get_kernel_map(features)
#
#     def save_feature_to_img(self):
#         # to numpy
#         feature = self.get_single_feature()
#         feature = feature.cpu().detach().numpy()
#         # # use sigmod to [0,1]
#         # feature = 1.0/(1+np.exp(-1*feature))
#         #
#         # # to [0,255]
#         # feature=np.round(feature*255)
#         # print(feature[0])
#         plt.imsave('./img.jpg', feature)

from PIL import Image
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch'
    def __init__(self, dataset_name, inputs, labels, transform=None):
        # 'Initialization'
        self.labels = labels
        # self.list_IDs = list_IDs
        self.inputs = inputs
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        # 'Denotes the total number of samples'
        return self.inputs.shape[0]

    def cutout(self, img, x, y, size):
        size = int(size/2)
        lx = np.maximum(0,x-size)
        rx = np.minimum(img.shape[0],x+size)
        ly = np.maximum(0, y - size)
        ry = np.minimum(img.shape[1], y + size)
        img[lx:rx,ly:ry,:] = 0
        return img

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]
        # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        img = self.inputs[index]

        if self.dataset_name == 'STL10' or self.dataset_name == 'TINY_IMAGENET':
            img = np.transpose(img, [1, 2, 0])

        # Cutout module begins
        # xcm = int(np.random.rand()*95)
        # ycm = int(np.random.rand()*95)
        # img = self.cutout(img,xcm,ycm,24)
        #Cutout module ends

        # img = np.float32(scipy.misc.imresize(img, 2.0))
        # img = Image.fromarray(np.uint8(img*255)).resize((28,28))

        # img = Image.fromarray(np.uint8(img*255))

        # Optional:
        # img = img / np.max(img)

        # img = color.rgb2gray(img)
        # img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)


        #
        # print(torch.max(img))
        # print(torc/h.min(img))



        # y = int(self.labels[index])
        y = self.labels[index]
        # y = self.transform(y)


        return img, y

from copy import copy

def load_dataset(dataset_name, val_splits, training_size):

    curr_dir = copy(os.getcwd())
    os.chdir('/MNIST/' + dataset_name)
    # os.chdir('../Common Datasets/' + dataset_name)
    # print('here in')
    a = os.listdir()

    listdict = []

    for split in range(val_splits):

        listdict.append(pickle.load(open(a[split], 'rb')))

        listdict[-1]['train_data'] = np.float32(listdict[-1]['train_data'][0:training_size, :, :])
        listdict[-1]['train_label'] = listdict[-1]['train_label'][0:training_size]

        # listdict[-1]['train_data'] = np.float32(listdict[-1]['data'][0:training_size, :, :])
        # listdict[-1]['train_label'] = listdict[-1]['label'][0:training_size]

        # for i in range(listdict[-1]['test_data'].shape[0]):
        #
        #     listdict[-1]['test_data'][i,0,:,:] = clipped_zoom(np.float32(listdict[-1]['test_data'][i,0,:,:]),zoom_factor=1.0/0.8,order=3)
        #     listdict[-1]['test_data'][i,1,:,:] = clipped_zoom(np.float32(listdict[-1]['test_data'][i,1,:,:]),zoom_factor=1.0/0.8,order=3)
        #     listdict[-1]['test_data'][i,2,:,:] = clipped_zoom(np.float32(listdict[-1]['test_data'][i,2,:,:]),zoom_factor=1.0/0.8,order=3)
        #


        # listdict[-1]['test_data'] = np.float32(listdict[-1]['test_data'])

        # listdict[-1]['test_label'] = np.float32(listdict[-1]['test_label'])


    os.chdir(curr_dir)

    return listdict