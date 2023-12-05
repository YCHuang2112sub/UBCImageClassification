#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !python script.py


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
# from torchvision import models
from torchsummary import summary
import PIL
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pandas as pd
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix


from pathlib import Path
from collections import defaultdict
from tqdm.notebook import tqdm
import validators

import argparse
import re
from collections import Counter

import logging


# In[ ]:





# In[3]:



print("number_of_cpus: ", torch.get_num_threads())
torch.set_num_threads(8)
print("confined to number_of_cpus: ", torch.get_num_threads())


# In[4]:


## using argparse to set parameters
parser = argparse.ArgumentParser(description='Train model on UCB Image Dataset')
parser.add_argument('--source_dataset_dir', type=str, default='/projectnb/cs640grp/materials/UBC-OCEAN_CS640', help='path to dataset')
parser.add_argument('--local_dataset_dir', type=str, default='../../../dataset', help='path to local dataset')
parser.add_argument('--model_dir', type=str, default='./model', help='path to tained model')
parser.add_argument('--experiment_name', type=str, default='exp_1', help='experiment name')

# parser.add_argument('--train_image_folder', type=str, default='img_size_512x512', help='training image folder name')
# parser.add_argument('--eval_image_folder', type=str, default='test_img_size_512x512', help='evaluate image folder name')
# # parser.add_argument('--train_image_folder', type=str, default='train_images_compressed_80', help='training image folder name')
# parser.add_argument('--image_input_size', type=str, default='(512, 512)', help='input image size')
parser.add_argument('--train_image_folder', type=str, default='img_size_2048x2048', help='training image folder name')
parser.add_argument('--eval_image_folder', type=str, default='test_img_size_2048x2048', help='evaluate image folder name')
parser.add_argument('--image_input_size', type=str, default='(2048, 2048)', help='input image size')

parser.add_argument('--batch_size', type=int, default=8, help='batch size')
# parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--eval_patience', type=int, default=20, help='patience for early stopping')


# In[5]:


setting = None
try:
    __IPYTHON__
    _in_ipython_session = True
    settings = parser.parse_args("")
except NameError:
    _in_ipython_session = False
    settings = parser.parse_args()

print("settings:", vars(settings))
# # save settings in to a log file
# with open(Path(settings.model_dir, settings.experiment_name, 'settings.txt'), 'w') as f:
#     print(vars(settings), file=f)


# In[6]:


# settings.image_input_size = (2048, 2048)
# settings.train_image_folder = "img_size_2048x2048"
# settings.test_image_folder = "test_img_size_2048x2048"


# In[7]:


image_input_size = eval(settings.image_input_size)
assert isinstance(image_input_size, tuple) and len(image_input_size) == 2, "image_input_size must be a tuple of length 2"

# # vars(settings)
# # print(image_input_size)


# In[8]:


PIL.Image.MAX_IMAGE_PIXELS = 933120000
# IMAGE_INPUT_SIZE = (2048, 2048)

IMAGE_INPUT_SIZE = image_input_size


# In[9]:


def create_dir_if_not_exist(dir):
    # if not os.path.exists(dir):
    #     os.makedirs(dir)
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)


# In[10]:


SOURCE_DATASET_DIR = settings.source_dataset_dir
# DATASET_PATH = "dataset"
# TRAIN_IMAGE_FOLDER = "train_images_compressed_80"
# TEST_IMAGE_FOLDER = "test_images_compressed_80"
# TRAIN_IMAGE_FOLDER = f"img_size_{IMAGE_INPUT_SIZE[0]}x{IMAGE_INPUT_SIZE[1]}"
TRAIN_IMAGE_FOLDER = settings.train_image_folder
EVAL_IMAGE_FOLDER = settings.eval_image_folder


# LOCAL_DATASET_DIR = "./dataset"
# MODEL_DIR = "./model/"
# EXPERIMENT_NAME = "exp_1"

LOCAL_DATASET_DIR = settings.local_dataset_dir
MODEL_DIR = settings.model_dir
EXPERIMENT_NAME = settings.experiment_name

sub_folder_name = f"img_size_{IMAGE_INPUT_SIZE[0]}x{IMAGE_INPUT_SIZE[1]}_lr_{settings.lr}__batch_size_{settings.batch_size}__num_epochs_{settings.num_epochs}__weight_decay_{settings.weight_decay}__eval_patience_{settings.eval_patience}"
sub_folder_name = re.sub(r"\.", "p", sub_folder_name)
MODEL_SAVE_DIR = Path(MODEL_DIR, EXPERIMENT_NAME, sub_folder_name)
RESULT_DIR = Path("./result", EXPERIMENT_NAME, sub_folder_name)
print("RESULT_DIR:", RESULT_DIR)

LOG_DIR = Path("./log", EXPERIMENT_NAME)
print("LOG_DIR:", LOG_DIR)
create_dir_if_not_exist(LOG_DIR)


# In[11]:


# # logging.basicConfig(level=logging.DEBUG)



# logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w',
logging.basicConfig(level=logging.DEBUG, filename=LOG_DIR/'log.txt', filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

# if __name__ == "__main__":
# 	logging.debug('debug')
# 	logging.info('info')
# 	logging.warning('warning')
# 	logging.error('error')
# 	logging.critical('critical')
# 	time.sleep(5)


# In[12]:


# lr = 0.001
# # momentum = 0.9
# weight_decay = 0.0001
# num_epochs = 20
# batch_size = 32

eval_patience = settings.eval_patience
lr = settings.lr
# momentum = settings.momentum
weight_decay = settings.weight_decay
num_epochs = settings.num_epochs
batch_size = settings.batch_size


# In[13]:


create_dir_if_not_exist(LOCAL_DATASET_DIR)


# In[14]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


# In[15]:


#pandas load data from csv
train_csv = None
test_csv = None
all_labels = None

if os.path.exists(Path(LOCAL_DATASET_DIR) / 'train.csv'):
    train_csv = pd.read_csv(Path(LOCAL_DATASET_DIR) / 'train.csv').to_numpy()
else:
    train_csv = pd.read_csv(Path(SOURCE_DATASET_DIR) / 'train.csv').to_numpy()

if os.path.exists(Path(LOCAL_DATASET_DIR) / 'test.csv'):
    test_csv = pd.read_csv(Path(LOCAL_DATASET_DIR) / 'test.csv').to_numpy()
else:
    test_csv = pd.read_csv(Path(SOURCE_DATASET_DIR) / 'test.csv').to_numpy()

# load npy
if os.path.exists(Path(LOCAL_DATASET_DIR) / 'all_labels.npy'):
    all_labels = np.load(Path(LOCAL_DATASET_DIR) / 'all_labels.npy')
else:
    all_labels = np.load(Path(SOURCE_DATASET_DIR) / 'all_labels.npy')



# In[16]:


dict_id_to_label = {i: label for i, label in enumerate(all_labels)}
dict_label_to_id = {label: i for i, label in enumerate(all_labels)}


print("dict_id_to_label:", sorted(dict_id_to_label.items()))


# In[17]:


def tran_csv_to_img_path_and_label(x_csv, data_path, image_folder, dict_label_to_id):
    x_data = []
    for i in range(len(x_csv)):
        #get img path
        img_name = str(x_csv[i][0]) + ".jpg"
        img_path = Path(data_path)  / image_folder / img_name
        # check image is exist
        if not img_path.exists():
            print(f"image {img_path} does not exist")
            continue

        x_data.append([img_path, dict_label_to_id[x_csv[i][1]]])
    return x_data


# In[18]:


train_image_path_and_label = tran_csv_to_img_path_and_label(train_csv, LOCAL_DATASET_DIR, TRAIN_IMAGE_FOLDER, dict_label_to_id)
test_image_path_and_label = tran_csv_to_img_path_and_label(test_csv, LOCAL_DATASET_DIR, EVAL_IMAGE_FOLDER, dict_label_to_id)

# # Random split
# # train_set, valid_set = train_test_split(train_image_path_and_label, test_size=0.2, random_state=42)
# train_set, valid_set = train_test_split(train_image_path_and_label, test_size=0.2, random_state=42, 
#                                         stratify=[x[1] for x in train_image_path_and_label])
train_set = train_image_path_and_label
valid_set = test_image_path_and_label

if EXPERIMENT_NAME == "efficientnet_b0__train_on_all_data":
    train_set = train_image_path_and_label + test_image_path_and_label

print("train set size:", len(train_set))    
print("valid set size:", len(valid_set))
# print("test set size:", len(test_set))

path_list, labels = zip(*train_set)
print("train set category distribution: \n\t", Counter(labels))

path_list, labels = zip(*valid_set)
print("train set category distribution: \n\t", Counter(labels))


# In[19]:


def show_img_by_path(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    print(img.size, type(img))
    img = transforms.Resize(IMAGE_INPUT_SIZE)(img)
    print(img.size, type(img))
    img = transforms.ToTensor()(img)
    print(img.shape)
    # print(img)
    
# show_img_by_path(train_set[0][0])


# In[20]:


from torchvision.io import read_image

class UBCDataset(Dataset):
    def __init__(self, img_path_and_label, transform=None, target_transform=None, random_add_single_value=False):
        self.data = img_path_and_label
        self.transform = transform
        self.target_transform = target_transform
        self.random_add_single_value = random_add_single_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image = read_image(img_path)
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        # print(np.max(image), np.min(image))

        # image_np = np.array(image)
        # ### white to black
        # mask = (image_np >= np.array([230, 230, 230])).all(axis=-1)
        # image_np[mask] = [0, 0, 0]
        # image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(torch.max(image), torch.min(image))

        img_downsampled = transforms.Resize((512, 512), antialias=True)(image)
        # print(img_downsampled.shape)
        img_10 = image[: , 1024-256:1024+256, 384-256:384+256]
        img_11 = image[: , 1024-256:1024+256, 1024-256:1024+256]
        img_12 = image[: , 1024-256:1024+256, 2048-384-256:2048-384+256]
        
        # if self.random_add_single_value:
        #     image += torch.randn(1) * 0.01
        
        return img_downsampled, label, img_10, img_11, img_12


# In[21]:


# put data into dataloader
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_INPUT_SIZE, antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(180),
    transforms.RandomAffine(180, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, interpolation=PIL.Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# v2.Compose([v2.Resize(256, antialias = True),
#                               v2.CenterCrop(224),
#                               v2.ToImage(),
#                               v2.ToDtype(torch.float32, scale = True),
#                               v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


test_transform = transforms.Compose([
    transforms.Resize(IMAGE_INPUT_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

train_dataset = UBCDataset(train_set, transform=train_transform, random_add_single_value=True)
valid_dataset = UBCDataset(valid_set, transform=test_transform)
# test_dataset = UBCDataset(test_set, transform=test_transform)


# In[22]:


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# In[23]:


for imgs, labels, imgs_10, imgs_11, imgs_12 in tqdm(valid_dataloader):
    # # print(x[0].shape, x[1].shape)
    
    
    # # imgs_downsampled = transforms.Resize((512, 512), antialias=True)(imgs)
    image_grid_orig_downsampled = torchvision.utils.make_grid(imgs[:8], nrow=8)

    # # print(imgs_downsampled.shape)

    # # imgs_10 = imgs[: ,: , 1024-256:1024+256, 384-256:384+256]
    # # imgs_11 = imgs[: ,: , 1024-256:1024+256, 1024-256:1024+256]
    # # imgs_12 = imgs[: ,: , 1024-256:1024+256, 2048-384-256:2048-384+256]
    # # # print(imgs.shape)
    # # print(imgs[0])
    
    image_grid_10 = torchvision.utils.make_grid(imgs_10[:8], nrow=8)
    image_grid_11 = torchvision.utils.make_grid(imgs_11[:8], nrow=8)
    image_grid_12 = torchvision.utils.make_grid(imgs_12[:8], nrow=8)
    plt.figure()
    plt.imshow(image_grid_orig_downsampled.permute(1, 2, 0).numpy())
    plt.figure()
    plt.imshow(image_grid_10.permute(1, 2, 0).numpy())
    plt.figure()
    plt.imshow(image_grid_11.permute(1, 2, 0).numpy())
    plt.figure()
    plt.imshow(image_grid_12.permute(1, 2, 0).numpy())
    break


# In[24]:


#show image grid
def show_image_grid(dataloader, num_of_images=16):
    imgs, labels, imgs_10, imgs_11, imgs_12 = next(iter(dataloader))

    # imgs_tmp = imgs.permute(0, 2, 3, 1)
    # # mask = (imgs_tmp == torch.tensor([255, 255, 255])).all(dim=3, keepdim=True)
    # mask = (imgs_tmp >= torch.tensor([100, 100, 100])).all(dim=3)
    # imgs_tmp[mask] = torch.tensor([0, 0, 0]).float()
    # imgs = imgs_tmp.permute(0, 3, 1, 2)

    image_grid = torchvision.utils.make_grid(imgs[:num_of_images], nrow=8)
    plt.imshow(image_grid.permute(1, 2, 0).numpy())
    # plt.show()
    print("labels:", labels[:num_of_images], [dict_id_to_label[label.item()] for label in labels[:num_of_images]])

# # show_image_grid(train_dataloader)
# # show_image_grid( DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0), 64)
# show_image_grid( DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=0), 64)


# In[25]:


# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_v2_l', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=True)

# print(dir(efficientnet.modules))
# print(efficientnet.classifier.fc.in_features)


# In[26]:


efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

if EXPERIMENT_NAME == "efficientnet_b0":
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
elif EXPERIMENT_NAME == "efficientnet_b4":
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
elif EXPERIMENT_NAME == "efficientnet_widese_b0":
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
elif EXPERIMENT_NAME == "efficientnet_widese_b4":
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=True)
# elif EXPERIMENT_NAME == "efficientnet_v2_l":
#     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_v2_l', pretrained=True)

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# efficientnet.eval().to(device)
# vars(efficientnet)


# In[27]:



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  classes

        # self.inc = InConv(in_channels, 64//4)
        # self.down1 = Down(64//4, 128//8)
        # self.down2 = Down(128//8, 256//8 //2)
        # self.down3 = Down(256//8 //2, 512//8 //4)
        # self.down4 = Down(512//8 //4, 512//8 //4)
        # self.up1 = Up(1024//8 //4, 256//8 //2)
        # self.up2 = Up(512//8  //2, 128//8)
        # self.up3 = Up(256//8 , 64//4)
        # self.up4 = Up(128//4, 64//4)
        # self.outc = OutConv(64//4, classes)

        self.inc = InConv(in_channels, 64//4)
        self.down1 = Down(64//4, 128//8)
        self.down2 = Down(128//8, 256//8)
        self.down3 = Down(256//8, 512//8)
        self.down4 = Down(512//8, 512//8)
        self.up1 = Up(1024//8, 256//8)
        self.up2 = Up(512//8, 128//8)
        self.up3 = Up(256//8, 64//4)
        self.up4 = Up(128//4, 64//4)
        self.outc = OutConv(64//4, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        # print("x.shape:", x.shape, "x3.shape:", x3.shape)
        x = self.up2(x, x3)
        # print("x.shape:", x.shape, "x2.shape:", x2.shape)
        x = self.up3(x, x2)
        # print("x.shape:", x.shape, "x1.shape:", x1.shape)
        x = self.up4(x, x1)
        # print("x.shape:", x.shape)
        x = self.outc(x)
        # return x, (x1, x2, x3, x4, x5)
        xs = (x1, x2, x3, x4, x5)
        return x, x1, x2, x3, x4, x5


# In[28]:


class Output4CombineModel(nn.Module):
    def __init__(self, efficientnet, unet):
        super(Output4CombineModel, self).__init__()

        # self.cnn_input_combine = torch.nn.Sequential(
        #             torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
        #             torch.nn.ReLU(),
        #             torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=True),
        #             torch.nn.ReLU(),
        #             torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True),
        #             torch.nn.ReLU()
        #         )
        
        self.unet = copy.deepcopy(unet)

        self.efficientnet = copy.deepcopy(efficientnet)
        self.classifier_in_feature_size = self.efficientnet.classifier.fc.in_features
        
        self.classifier_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=self.classifier_in_feature_size, out_features=256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=256, out_features=5, bias=True),
            # torch.nn.Linear(in_features=256, out_features=16, bias=True),
        )
        self.efficientnet.classifier = self.classifier_head


        # self.combined_classifier_head = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=16*4, out_features=32, bias=True),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2, inplace=False),
        #     torch.nn.Linear(in_features=32, out_features=5, bias=True),
        # )

        self.transform_downsampler = transforms.Resize((512, 512), antialias=True)

        self.unet_params = []
        for name, params in self.unet.named_parameters():
            self.unet_params.append(params)

        self.feature_extractor_params = []
        for name, params in self.efficientnet.named_parameters():
            if(name[:10] == "classifier"):
                continue
            self.feature_extractor_params.append(params)

        self.classifier_head_params = []
        for name, params in self.classifier_head.named_parameters():
            self.classifier_head_params.append(params)

        # self.combined_classifier_head_params = []
        # for name, params in self.combined_classifier_head.named_parameters():
        #     self.combined_classifier_head_params.append(params)

    def forward(self, imgs, imgs_10, imgs_11, imgs_12):
        # imgs_downsampled = self.transform_downsampler(imgs)
        concat_in = torch.concat([imgs, imgs_10, imgs_11, imgs_12], dim=1)
        # combine_in = self.cnn_input_combine(concat_in)
        # combine_in = self.unet(concat_in)
        combine_in, x1, x2, x3, x4, x5 = self.unet(concat_in)
        # x1, x2, x3, x4, x5 = xs
        outputs = self.efficientnet(combine_in)
        # f_orig = self.efficientnet(imgs)
        # f_10 = self.efficientnet(imgs_10)
        # f_11 = self.efficientnet(imgs_11)
        # f_12 = self.efficientnet(imgs_12)
        # comb_f = torch.concat([f_orig, f_10, f_11, f_12], dim=1)
        # print(comb_f.shape)
        # outputs = self.combined_classifier_head(comb_f)
        
        
        return outputs, combine_in
            


# In[29]:


# model_raw = efficientnet
# classifier_in_feature_size = model_raw.classifier.fc.in_features
# classifier_head = torch.nn.Sequential(
#     torch.nn.AdaptiveAvgPool2d(output_size=1),
#     torch.nn.Flatten(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=classifier_in_feature_size, out_features=256, bias=True),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     # torch.nn.Linear(in_features=256, out_features=5, bias=True),
#     torch.nn.Linear(in_features=256, out_features=16, bias=True),
# )
# model_raw.classifier = classifier_head
# model_raw = model_raw.to(device)



# combined_classifier_head = torch.nn.Sequential(
#     torch.nn.Linear(in_features=16*4, out_features=32, bias=True),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=32, out_features=5, bias=True),
# )
# combined_classifier_head = combined_classifier_head.to(device)




# feature_extractor_params = []
# for name, params in model_raw.named_parameters():
#     if(name[:10] == "classifier"):
#         continue
#     feature_extractor_params.append(params)

# classifier_head_params = []
# for name, params in classifier_head.named_parameters():
#     classifier_head_params.append(params)

# combined_classifier_head_params = []
# for name, params in combined_classifier_head.named_parameters():
#     combined_classifier_head_params.append(params)

unet = Unet(3*4, 3)

model_raw = Output4CombineModel(efficientnet, unet)
model_raw = model_raw.to(device)
unet_params = model_raw.unet_params
feature_extractor_params = model_raw.feature_extractor_params
classifier_head_params = model_raw.classifier_head_params
# combined_classifier_head_params = model_raw.combined_classifier_head_params


# In[30]:


# print("feature_extractor_params:", len(feature_extractor_params))
# print(classifier_head_params)


# In[31]:


# summary(model_raw, (3, ) + IMAGE_INPUT_SIZE, device=device.type)
# summary(model_raw, [(3, ) + (512, 512), (3, ) + (512, 512), (3, ) + (512, 512), (3, ) + (512, 512)], device=device.type)


# In[32]:



criteria = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_raw.classifier.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = optim.Adam([{ "params":unet_params, "lr":lr},
                        {"params":feature_extractor_params, "lr":1e-7}, 
                        {"params":classifier_head_params, "lr":lr},
                        # {"params":combined_classifier_head_params, "lr":lr}
                        ], 
                       lr=lr, weight_decay=weight_decay)


# In[33]:




def get_category_accuracy(y_gt: np.array, y_pred: np.array, n_category):
    # category_accuracy = np.zeros(n_category)

    assert(len(y_gt) == len(y_pred))
    assert(len(y_gt.shape) == 1 and len(y_pred.shape) == 1)

    cat_mask_2d = (y_gt == np.arange(n_category).reshape(-1, 1))
    category_accuracy = ((y_gt == y_pred) * cat_mask_2d).sum(axis=1) / cat_mask_2d.sum(axis=1)

    return category_accuracy

# a = np.random.randint(0, 5, 10)
# b = np.random.randint(0, 5, 10)
# print("a:", a)
# print("b:", b)

# # print("get_category_accuracy:", get_category_accuracy(y_gt=a, y_pred=b, n_category=5))
# category_accuracy = get_category_accuracy(y_gt=np.array(a), y_pred=np.array(b), n_category=5)
# print("category_accuracy:", category_accuracy)
# print(str(category_accuracy))
# print(type(category_accuracy[0]))
# confusion_matrix_result = confusion_matrix(y_true=a, y_pred=b, labels=[0, 1, 2, 3, 4])
# print("get_category_accuracy:" + str(category_accuracy))


# In[34]:


def evaluation(model, valid_dataloader, criteria, device):
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0

        y_gt = []
        y_pred = []

        # for imgs, labels in tqdm(valid_dataloader):
        for imgs, labels, imgs_10, imgs_11, imgs_12 in tqdm(valid_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs_10 = imgs_10.to(device)
            imgs_11 = imgs_11.to(device)
            imgs_12 = imgs_12.to(device)

            # imgs_downsampled = transforms.Resize((512, 512), antialias=True)(imgs)
            # imgs_10 = imgs[: ,: , 1024-256:1024+256, 384-256:384+256]
            # imgs_11 = imgs[: ,: , 1024-256:1024+256, 1024-256:1024+256]
            # imgs_12 = imgs[: ,: , 1024-256:1024+256, 2048-384-256:2048-384+256]
            # # imgs_in = torch.concat([imgs_downsampled, imgs_10, imgs_11, imgs_12], dim=1)

            # outputs_orig = model(imgs_downsampled)
            # outputs_orig = model(imgs)
            # outputs_10 = model(imgs_10)
            # outputs_11 = model(imgs_11)
            # outputs_12 = model(imgs_12)
            

            with torch.no_grad():
                outputs, combine_in = model(imgs, imgs_10, imgs_11, imgs_12)

                # similarity_scores_a00 = torch.matmul(f_orig, f_10.T)
                # similarity_scores_a01 = torch.matmul(f_orig, f_11.T)
                # similarity_scores_a02 = torch.matmul(f_orig, f_12.T)
                # similarity_scores_a03 = torch.matmul(f_10, f_11.T)
                # similarity_scores_a04 = torch.matmul(f_10, f_11.T)
                # similarity_scores_a05 = torch.matmul(f_11, f_12.T)

                # gt_similarity = torch.eye(similarity_scores_a00.shape[0], device=similarity_scores_a00.device)
                # loss_similarity = criteria(similarity_scores_a00, gt_similarity) + \
                #                     criteria(similarity_scores_a01, gt_similarity) + \
                #                     criteria(similarity_scores_a02, gt_similarity) + \
                #                     criteria(similarity_scores_a03, gt_similarity) + \
                #                     criteria(similarity_scores_a04, gt_similarity) + \
                #                     criteria(similarity_scores_a05, gt_similarity)    


                _, preds = torch.max(outputs, -1)
                loss_classification = criteria(outputs, labels)

                # loss = loss_classification + loss_similarity / 6.0
                loss = loss_classification

                # # outputs = model(imgs_in)
                # # outputs = model(imgs)
                # _, preds = torch.max(outputs, -1)
                # loss = criteria(outputs, labels)

            valid_loss += loss.item() * imgs.size(0)
            valid_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()

            y_gt.extend(labels.data.cpu().numpy().reshape(-1))
            y_pred.extend(preds.cpu().numpy().reshape(-1))


        # print(y_gt)
        # print(y_pred)
        confusion_matrix_result = confusion_matrix(y_gt, y_pred)
        print("confusion_matrix_result:\n" + str(confusion_matrix_result))
        logging.info("confusion_matrix_result:\n" + str(confusion_matrix_result))

        category_accuracy = get_category_accuracy(y_gt=np.array(y_gt), y_pred=np.array(y_pred), n_category=5)
        print("get_category_accuracy:" + str(category_accuracy))
        logging.info("get_category_accuracy:" + str(category_accuracy))
        
        valid_loss = valid_loss / len(valid_dataloader.dataset)
        valid_acc = valid_corrects / len(valid_dataloader.dataset)

        valid_balanced_acc = balanced_accuracy_score(y_gt, y_pred)

        return valid_loss, valid_acc, valid_balanced_acc, y_pred, y_gt


# In[35]:


def train(model, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, eval_patience, device):
    train_loss_list = []
    train_acc_list = []
    train_balanced_acc_list = []

    valid_loss_list = []
    valid_acc_list = []
    valid_balanced_acc_list = []

    best_valid_loss = float('inf')
    best_valid_acc = -0.0001
    best_valid_balanced_acc = -0.0001

    best_model_valid_loss = None
    best_model_valid_acc = None
    best_model_valid_balanced_acc = None

    start_time = time.time()

    counter_eval_not_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        logging.info('-' * 10)

        model.train()
        train_loss = 0.0
        train_corrects = 0
        
        y_gt = []
        y_pred = []

        # for imgs, labels in tqdm(train_dataloader):
        for imgs, labels, imgs_10, imgs_11, imgs_12  in tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs_10 = imgs_10.to(device)
            imgs_11 = imgs_11.to(device)
            imgs_12 = imgs_12.to(device)


    
            # imgs_downsampled = transforms.Resize((512, 512), antialias=True)(imgs)
            # imgs_10 = imgs[: ,: , 1024-256:1024+256, 384-256:384+256]
            # imgs_11 = imgs[: ,: , 1024-256:1024+256, 1024-256:1024+256]
            # imgs_12 = imgs[: ,: , 1024-256:1024+256, 2048-384-256:2048-384+256]
            # imgs_in = torch.concat([imgs_downsampled, imgs_10, imgs_11, imgs_12], dim=1)

            # # outputs_orig = model(imgs_downsampled)
            # outputs_orig = model(imgs)
            # outputs_10 = model(imgs_10)
            # outputs_11 = model(imgs_11)
            # outputs_12 = model(imgs_12)
            
            outputs, combine_in = model(imgs, imgs_10, imgs_11, imgs_12)
            # outputs = outputs_orig
            # outputs = model(imgs)

            # similarity_scores_a00 = torch.matmul(f_orig, f_10.T)
            # similarity_scores_a01 = torch.matmul(f_orig, f_11.T)
            # similarity_scores_a02 = torch.matmul(f_orig, f_12.T)
            # similarity_scores_a03 = torch.matmul(f_10, f_11.T)
            # similarity_scores_a04 = torch.matmul(f_10, f_11.T)
            # similarity_scores_a05 = torch.matmul(f_11, f_12.T)

            # gt_similarity = torch.eye(similarity_scores_a00.shape[0], device=similarity_scores_a00.device)
            # loss_similarity = criteria(similarity_scores_a00, gt_similarity) + \
            #                     criteria(similarity_scores_a01, gt_similarity) + \
            #                     criteria(similarity_scores_a02, gt_similarity) + \
            #                     criteria(similarity_scores_a03, gt_similarity) + \
            #                     criteria(similarity_scores_a04, gt_similarity) + \
            #                     criteria(similarity_scores_a05, gt_similarity)    


            _, preds = torch.max(outputs, -1)
            loss_classification = criteria(outputs, labels)

            loss = loss_classification
            # loss = loss_classification + loss_similarity / 6.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()

            y_gt.extend(labels.data.cpu().numpy().reshape(-1))
            y_pred.extend(preds.cpu().numpy().reshape(-1))


        # print(y_gt)
        # print(y_pred)
        confusion_matrix_result = confusion_matrix(y_gt, y_pred)
        print("confusion_matrix_result:\n" + str(confusion_matrix_result))
        logging.info("confusion_matrix_result:\n" + str(confusion_matrix_result))

        category_accuracy = get_category_accuracy(y_gt=np.array(y_gt), y_pred=np.array(y_pred), n_category=5)
        print("get_category_accuracy:" + str(category_accuracy))
        logging.info("get_category_accuracy:" + str(category_accuracy))
        
        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects / len(train_dataloader.dataset)
        train_balanced_acc = balanced_accuracy_score(y_gt, y_pred)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_balanced_acc_list.append(train_balanced_acc)

        print(f'Train loss: {train_loss:.4f} Acc: {train_acc:.4f} Balanced Acc: {train_balanced_acc:.4f}')
        logging.info(f'Train loss: {train_loss:.4f} Acc: {train_acc:.4f} Balanced Acc: {train_balanced_acc:.4f}')
        print("id_to_label:", sorted(dict_id_to_label.items()))
        logging.info("id_to_label:" + str(sorted(dict_id_to_label.items())))
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
        logging.info(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        valid_loss, valid_acc, valid_balanced_acc, _, _ = evaluation(model, valid_dataloader, criteria, device)

        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        valid_balanced_acc_list.append(valid_balanced_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_valid_loss = copy.deepcopy(model)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_valid_acc = copy.deepcopy(model)
        # else:
        #     counter_eval_not_improve += 1

        if valid_balanced_acc > best_valid_balanced_acc:
            best_valid_balanced_acc = valid_balanced_acc
            best_model_valid_balanced_acc = copy.deepcopy(model)
        else:
            counter_eval_not_improve += 1


        print(f'Valid loss: {valid_loss:.4f} Acc: {valid_acc:.4f} Balanced Acc: {valid_balanced_acc:.4f}')
        logging.info(f'Valid loss: {valid_loss:.4f} Acc: {valid_acc:.4f} Balanced Acc: {valid_balanced_acc:.4f}')
        print("id_to_label:", sorted(dict_id_to_label.items()))
        logging.info("id_to_label:" + str(sorted(dict_id_to_label.items())))
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n\n')
        logging.info(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}\n\n')

        if counter_eval_not_improve >= eval_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            logging.info(f'Early stopping at epoch {epoch + 1}')
            break
        else:
            counter_eval_not_improve = 0

    return model, best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc,            train_loss_list, train_acc_list, train_balanced_acc_list,            valid_loss_list, valid_acc_list, valid_balanced_acc_list,            best_valid_loss, best_valid_acc


# In[36]:


def store_result(best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc,                  train_loss_list, train_acc_list, train_balanced_acc_list,                  valid_loss_list, valid_acc_list, valid_balanced_acc_list):
    create_dir_if_not_exist(MODEL_SAVE_DIR)
    create_dir_if_not_exist(RESULT_DIR)

    torch.save(best_model_valid_acc.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_acc.pth")
    torch.save(best_model_valid_loss.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_loss.pth")
    torch.save(best_model_valid_balanced_acc.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_balanced_acc.pth")

    with open(Path(RESULT_DIR) / "train_loss_list.pkl", "wb") as f:
        pickle.dump(train_loss_list, f)
    with open(Path(RESULT_DIR) / "train_acc_list.pkl", "wb") as f:
        pickle.dump(train_acc_list, f)
    with open(Path(RESULT_DIR) / "train_balanced_acc_list.pkl", "wb") as f:
        pickle.dump(train_balanced_acc_list, f)

    with open(Path(RESULT_DIR) / "valid_loss_list.pkl", "wb") as f:
        pickle.dump(valid_loss_list, f)
    with open(Path(RESULT_DIR) / "valid_acc_list.pkl", "wb") as f:
        pickle.dump(valid_acc_list, f)
    with open(Path(RESULT_DIR) / "valid_balanced_acc_list.pkl", "wb") as f:
        pickle.dump(valid_balanced_acc_list, f)


# In[37]:


def plot_train_eval_result(
           train_loss_list, train_acc_list, train_balanced_acc_list, \
           valid_loss_list, valid_acc_list, valid_balanced_acc_list):
    epochs = np.arange(1, len(train_loss_list) + 1)

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss_list, label='train')
    plt.plot(epochs, valid_loss_list, label='valid')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(3, 1, 2)
    plt.plot(epochs, [x*100 for x in train_acc_list], label='train')
    plt.plot(epochs, [x*100 for x in valid_acc_list], label='valid')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.subplot(3, 1, 3)
    plt.plot(epochs, [x*100 for x in train_balanced_acc_list], label='train')
    plt.plot(epochs, [x*100 for x in valid_balanced_acc_list], label='valid')
    plt.title('Balanced Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy (%)')

    plt.tight_layout()


# In[38]:


model_trained, best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc, train_loss_list, train_acc_list, train_balanced_acc_list, valid_loss_list, valid_acc_list, valid_balanced_acc_list, best_valid_loss, best_valid_acc = train(model_raw, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, eval_patience, device)

print("best_valid_loss:", np.min(valid_loss_list))
print("best_valid_acc:", np.max(valid_acc_list))
print("best_valid_balanced_acc:", np.max(valid_balanced_acc_list))


# In[ ]:


# # train_loss_list = [data.cpu().item() for data in train_loss_list]
# train_acc_list = [data.cpu().item() for data in train_acc_list]
# # valid_loss_list = [data.cpu().item() for data in valid_loss_list]
# valid_acc_list = [data.cpu().item() for data in valid_acc_list]


# In[ ]:


store_result(best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc,              train_loss_list, train_acc_list, train_balanced_acc_list,              valid_loss_list, valid_acc_list, valid_balanced_acc_list)


# In[ ]:


plot_train_eval_result(
           train_loss_list, train_acc_list, train_balanced_acc_list, \
           valid_loss_list, valid_acc_list, valid_balanced_acc_list)


# In[ ]:





# In[ ]:


valid_loss, valid_acc, valid_balanced_acc, y_pred, y_gt = evaluation(best_model_valid_acc, valid_dataloader, criteria, device)

print("valid_loss:", valid_loss, "valid_acc:", valid_acc, "valid_balanced_acc:", valid_balanced_acc)

path_list, labels = zip(*valid_set)
for pred, gt, label, path in zip(y_pred, y_gt, labels, path_list):
    # if pred != gt:
    # if gt != label:
    filename = path.name
    print(f"pred: {dict_id_to_label[pred]}, gt: {dict_id_to_label[gt]}, label: {dict_id_to_label[label]}, filename: {filename}")


# In[ ]:


print(f"pred: {dict_id_to_label[pred]}, gt: {dict_id_to_label[gt]}, label: {dict_id_to_label[label]}, filename: {filename}")
# put pred, gt, label, filename into a dataframe
pd_predict = pd.DataFrame({"image_id": [path.name.split(".")[0] for path in path_list], "label": y_pred, "gt": y_gt})
# display(pd_predict)
pd_predict.to_csv(Path(RESULT_DIR) / "pd_predict.csv", index=False)


# 

# In[ ]:


# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# efficientnet.eval().to(device)


# In[ ]:


# # Download an example image
# import urllib
# url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png", "TCGA_CS_4944.png")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)


# In[ ]:


# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#     in_channels=3, out_channels=1, init_features=32, pretrained=True)


# In[ ]:


# import numpy as np
# from PIL import Image
# from torchvision import transforms

# input_image = Image.open(filename)
# m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=m, std=s),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)

# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model = model.to('cuda')

# with torch.no_grad():
#     output = model(input_batch)

# print(torch.round(output[0]))


# In[ ]:




