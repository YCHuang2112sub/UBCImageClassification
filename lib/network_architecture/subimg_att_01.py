import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
# from torchvision import models
from torchsummary import summary
# import PIL
# from PIL import Image

# import numpy as np


class Downscale4StageCNN(nn.Module):
    def __init__(self, in_ch=3, channel_sizes=[16,32,64,128]):
        super(Downscale4StageCNN, self).__init__()

        assert len(channel_sizes) == 4
        # assert channel_sizes[-1] == out_ch
        
        ch_1, ch_2, ch_3, ch_4 = channel_sizes

        self.conv1 = nn.Conv2d(in_ch, ch_1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_1)
        
        self.conv2 = nn.Conv2d(ch_1, ch_2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_2)
        
        self.conv3 = nn.Conv2d(ch_2, ch_3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(ch_3)
        
        self.conv4 = nn.Conv2d(ch_3, ch_4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(ch_4)

        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.max_pooling(x1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.max_pooling(x2)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
    
        x4 = self.max_pooling(x3)
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
    
        return x4, (x1, x2, x3, x4)


class ImgAttClassifier(nn.Module):
    def __init__(self, subimage_size = (16, 16)):
        super(ImgAttClassifier, self).__init__()

        self.subimage_size = subimage_size

        # masked_filter = k_d4_sub
        self.relu = nn.ReLU()

        self.linear1 = nn.Linear(256, 256//4)
        self.conv1 = nn.Conv2d(3072, 3072//4, (1,1), padding=0)
        self.conv2 = nn.Conv2d(3072//4, 3072//16, (8,8), padding=0)
        self.linear2 = nn.Linear(3072//16, 128)
        self.linear3 = nn.Linear(128, 5)



    def forward(self, imgs, d4):
        imgs_sub = self.split2SubImages(imgs, self.subimage_size) # B * num_imgs * C * Ho * Wo
        B, num_imgs, C, Ho, Wo = imgs_sub.shape
        k_imgs_sub = imgs_sub.reshape((B, num_imgs*C, Ho*Wo))
        # print(k_imgs_sub.shape)

        # print("d4_sub:")
        imgs_d4_sub = self.split2SubImages(d4, self.subimage_size) # B * num_imgs * C * Ho * Wo
        B, num_imgs, C, Ho, Wo = imgs_d4_sub.shape
        k_d4_sub = imgs_d4_sub.reshape((B, num_imgs*C, Ho*Wo))
        # print(k_d4_sub.shape)

        assert(k_d4_sub.shape == k_imgs_sub.shape)

        
        masked_filter = self.relu(k_d4_sub)
        k_filtered_imgs = k_imgs_sub * masked_filter

        B2, C2, embed_dim2 = k_filtered_imgs.shape
        k_filtered_imgs_indep = self.linear1(k_filtered_imgs)
        filtered_imgs_indep = k_filtered_imgs_indep.reshape(B2, C2, 8, 8)
        filtered_imgs_cross = self.conv1(filtered_imgs_indep)
        filtered_imgs = self.conv2(filtered_imgs_cross)
        out_1 = torch.flatten(filtered_imgs, start_dim=1)
        out_1 = self.relu(out_1)
        out_2 = self.linear2(out_1)
        out_2 = self.relu(out_2)
        out_3 = self.linear3(out_2)
        
        
        return out_3

    
    def split2SubImages(self, imgs: torch.tensor, subimage_size: tuple):
        imgs_shape = imgs.shape
        assert(len(imgs_shape) == 4) # B * C * Hi * Wi 
        assert(len(subimage_size) == 2) # Ho * Wo
        # check fully divisible
        assert( (imgs_shape[2] % subimage_size[0] == 0) and (imgs_shape[3] % subimage_size[1] == 0))
        
        a1 = torch.nn.Unfold(subimage_size, dilation=(1,1), stride=subimage_size)(imgs)

        B = imgs.shape[0]
        C = imgs.shape[1]
        L = a1.shape[-1]
        a2 = a1.permute(0,2,1) # B * L * (C x Ho x Wo)
        a3 = a2.reshape((B,L,C)+subimage_size) 

        # out_shape = a3.shape # B * num_imgs * C * Ho * Wo

        return a3 # B * num_imgs * C * Ho * Wo