import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms

# from torch.utils.data import Dataset, DataLoader
# # from torchvision import models
# from torchsummary import summary
# import PIL
# from PIL import Image

# import numpy as np


def split2SubImages(imgs: torch.tensor, subimage_size: tuple):
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