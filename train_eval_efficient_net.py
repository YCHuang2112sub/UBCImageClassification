#!/usr/bin/env python
# coding: utf-8

# In[21]:





# In[22]:


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

from pathlib import Path
from collections import defaultdict
from tqdm.notebook import tqdm
import validators


# In[2]:


PIL.Image.MAX_IMAGE_PIXELS = 933120000
# IMAGE_INPUT_SIZE = (2048, 2048)
IMAGE_INPUT_SIZE = (256, 256)


# In[3]:


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# In[4]:


SOURCE_DATASET_DIR = "/projectnb/cs640grp/materials/UBC-OCEAN_CS640"
# DATASET_PATH = "dataset"
# TRAIN_IMAGE_FOLDER = "train_images_compressed_80"
# TEST_IMAGE_FOLDER = "test_images_compressed_80"
TRAIN_IMAGE_FOLDER = f"img_size_{IMAGE_INPUT_SIZE[0]}x{IMAGE_INPUT_SIZE[1]}"


LOCAL_DATASET_DIR = "./dataset"

MODEL_DIR = "./model/"

EXPERIMENT_NAME = "exp_1"
MODEL_SAVE_DIR = Path(MODEL_DIR, EXPERIMENT_NAME)
RESULT_DIR = Path("./result", EXPERIMENT_NAME)



# In[5]:


create_dir_if_not_exist(LOCAL_DATASET_DIR)


# In[6]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


# In[7]:


#show Pahtlib  combine two path example
# print(Path(DATASET_PATH) / "train")


# In[8]:


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




# In[ ]:





# In[9]:


# print(train_csv.shape)
# print(test_csv.shape)
# print(all_labels.shape)


# In[10]:


dict_id_to_label = {i: label for i, label in enumerate(all_labels)}
dict_label_to_id = {label: i for i, label in enumerate(all_labels)}


# In[11]:


def tran_csv_to_img_path_and_label(x_csv, data_path, image_folder, dict_label_to_id):
    x_data = []
    for i in range(len(x_csv)):
        #get img path
        img_name = str(x_csv[i][0]) + ".jpg"
        img_path = Path(data_path)  / image_folder / img_name
        # check image is exist
        if not img_path.exists():
            continue

        x_data.append([img_path, dict_label_to_id[x_csv[i][1]]])
    return x_data


# In[12]:


# def tran_csv_to_data(x_csv, data_path, image_folder, dict_label_to_id):
#     x_data = []
#     for i in range(len(x_csv)):
#         #get img path
#         img_name = str(x_csv[i][0]) + ".jpg"
#         img_path = Path(data_path)  / image_folder / img_name
#         # check image is exist
#         if not img_path.exists():
#             continue
#         #load image
#         img = Image.open(img_path)

#         x_data.append([img, dict_label_to_id[x_csv[i][1]]])
#     return x_data


# In[13]:


train_image_path_and_label = tran_csv_to_img_path_and_label(train_csv, LOCAL_DATASET_DIR, TRAIN_IMAGE_FOLDER, dict_label_to_id)
# test_image_path_and_label = tran_csv_to_img_path_and_label(test_csv, LOCAL_DATASET_DIR, TEST_IMAGE_FOLDER, dict_label_to_id)

# Random split
train_set, valid_set = train_test_split(train_image_path_and_label, test_size=0.2, random_state=42)
# test_set = test_image_path_and_label


print("train set size:", len(train_set))    
print("valid set size:", len(valid_set))
# print("test set size:", len(test_set))


# In[14]:


x = Image.open(train_set[0][0])
plt.imshow(x.resize(IMAGE_INPUT_SIZE))
# plt.imshow(x.resize((1024, 1024)))
print(x.size, type(x))
x = transforms.Resize(IMAGE_INPUT_SIZE)(x)
print(x.size, type(x))
x = transforms.ToTensor()(x)
print(x.shape)
# x.save("test.jpg")


# In[15]:


from torchvision.io import read_image

class UBCDataset(Dataset):
    def __init__(self, img_path_and_label, transform=None, target_transform=None):
        self.data = img_path_and_label
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # image = read_image(img_path)
        image_path, label = self.data[idx]
        image = Image.open(image_path)
        # print(np.max(image), np.min(image))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # print(torch.max(image), torch.min(image))
        return image, label


# In[16]:


# # put data into UBCDataset
# train_dataset = UBCDataset(train_set, transform=transforms.ToTensor())
# valid_dataset = UBCDataset(valid_set, transform=transforms.ToTensor())
# test_dataset = UBCDataset(test_set, transform=transforms.ToTensor())


# In[17]:


# put data into dataloader
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = UBCDataset(train_set, transform=train_transform)
valid_dataset = UBCDataset(valid_set, transform=test_transform)
# test_dataset = UBCDataset(test_set, transform=test_transform)



# In[18]:


train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# In[19]:


for imgs, labels in train_dataloader:
    break
print(labels)
print(imgs.shape)
image_grid = torchvision.utils.make_grid(imgs, nrow=8)
plt.imshow(image_grid.permute(1, 2, 0).numpy())


# In[26]:


efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

# efficientnet.eval().to(device)
# vars(efficientnet)


# In[27]:


model_raw = efficientnet
model_raw.classifier = torch.nn.Sequential(
    torch.nn.AdaptiveAvgPool2d(output_size=1),
    torch.nn.Flatten(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(in_features=1280, out_features=1000, bias=True)
)
model_raw = model_raw.to(device)


# In[28]:


summary(model_raw, (3, ) + IMAGE_INPUT_SIZE, device=device.type)


# In[29]:


lr = 0.001
# momentum = 0.9
weight_decay = 0.0001
num_epochs = 20
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_raw.parameters(), lr=lr, weight_decay=weight_decay)



# In[30]:


def eval(model, valid_dataloader, criteria, device):
        model.eval()
        valid_loss = 0.0
        valid_corrects = 0

        for imgs, labels in tqdm(valid_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(imgs)
                _, preds = torch.max(outputs, -1)
                loss = criteria(outputs, labels)

            valid_loss += loss.item() * imgs.size(0)
            valid_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()

        valid_loss = valid_loss / len(valid_dataloader.dataset)
        valid_acc = valid_corrects / len(valid_dataloader.dataset)

        return valid_loss, valid_acc


# In[31]:


def train(model, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, device):
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    best_model_valid_loss = None
    best_model_valid_acc = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        model.train()
        train_loss = 0.0
        train_corrects = 0

        for imgs, labels in tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)


            outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            loss = criteria(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()

        train_loss = train_loss / len(train_dataloader.dataset)
        train_acc = train_corrects / len(train_dataloader.dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        print(f'Train loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        valid_loss, valid_acc = eval(model, valid_dataloader, criteria, device)

        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_valid_loss = copy.deepcopy(model)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_valid_acc = copy.deepcopy(model)

        print(f'Valid loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')

    return model, best_model_valid_acc, best_model_valid_loss, \
           train_loss_list, train_acc_list, valid_loss_list, valid_acc_list


# In[32]:


def store_result(best_model_valid_acc, best_model_valid_loss, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list):
    create_dir_if_not_exist(MODEL_SAVE_DIR)
    create_dir_if_not_exist(RESULT_DIR)

    torch.save(best_model_valid_acc.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_acc.pth")
    torch.save(best_model_valid_loss.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_loss.pth")

    with open(Path(RESULT_DIR) / "train_loss_list.pkl", "wb") as f:
        pickle.dump(train_loss_list, f)
    with open(Path(RESULT_DIR) / "train_acc_list.pkl", "wb") as f:
        pickle.dump(train_acc_list, f)
    with open(Path(RESULT_DIR) / "valid_loss_list.pkl", "wb") as f:
        pickle.dump(valid_loss_list, f)
    with open(Path(RESULT_DIR) / "valid_acc_list.pkl", "wb") as f:
        pickle.dump(valid_acc_list, f)


# In[33]:


def plot_train_eval_result(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list):
    epochs = np.arange(1, len(train_loss_list) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss_list, label='train')
    plt.plot(epochs, valid_loss_list, label='valid')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, [x*100 for x in train_acc_list], label='train')
    plt.plot(epochs, [x*100 for x in valid_acc_list], label='valid')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()


# In[34]:


model_trained, best_model_valid_acc, best_model_valid_loss, \
train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = \
train(model_raw, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, device)



# In[ ]:


# # train_loss_list = [data.cpu().item() for data in train_loss_list]
# train_acc_list = [data.cpu().item() for data in train_acc_list]
# # valid_loss_list = [data.cpu().item() for data in valid_loss_list]
# valid_acc_list = [data.cpu().item() for data in valid_acc_list]


# In[ ]:


store_result(best_model_valid_acc, best_model_valid_loss, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list)


# In[ ]:


plot_train_eval_result(train_loss_list, train_acc_list, valid_loss_list, valid_acc_list)


# In[ ]:





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

