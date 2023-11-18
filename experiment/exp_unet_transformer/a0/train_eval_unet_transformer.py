#!/usr/bin/env python
# coding: utf-8

# In[74]:


# !python script.py


# In[75]:


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


import sys
import os
# sys.path.append(Path(os.getcwd(),'../../').absolute())
# print(type(Path(os.getcwd(),'../../').absolute()))

# try:
#     __IPYTHON__
sys.path.append(os.getcwd() + '/../../../')
sys.path.append(os.getcwd() + '/../../../../')
# except NameError:
#     _in_ipython_session = False
#     sys.path.append(os.path.realpath(__file__) + '/../../../')
# print(os.path.realpath(__file__))
# print(sys.path)
from lib.network_architecture.unet_transformer_01 import MyViTBlock, FeatureTransformer, Unet, BridgingModel,                                                          get_unet_transformer_model_output

import logging


# In[76]:



print("number_of_cpus: ", torch.get_num_threads())
torch.set_num_threads(16)
print("confined to number_of_cpus: ", torch.get_num_threads())


# In[77]:


## using argparse to set parameters
parser = argparse.ArgumentParser(description='Train model on UCB Image Dataset')
parser.add_argument('--source_dataset_dir', type=str, default='/projectnb/cs640grp/materials/UBC-OCEAN_CS640', help='path to dataset')
parser.add_argument('--local_dataset_dir', type=str, default='../../../dataset', help='path to local dataset')
parser.add_argument('--model_dir', type=str, default='./model', help='path to tained model')
parser.add_argument('--experiment_name', type=str, default='exp_1', help='experiment name')

parser.add_argument('--train_image_folder', type=str, default='img_size_512x512', help='training image folder name')
parser.add_argument('--eval_image_folder', type=str, default='test_img_size_512x512', help='evaluate image folder name')
# parser.add_argument('--train_image_folder', type=str, default='train_images_compressed_80', help='training image folder name')
parser.add_argument('--image_input_size', type=str, default='(512, 512)', help='input image size')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--eval_patience', type=int, default=20, help='patience for early stopping')


# In[78]:


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


# In[79]:


image_input_size = eval(settings.image_input_size)
assert isinstance(image_input_size, tuple) and len(image_input_size) == 2, "image_input_size must be a tuple of length 2"

# # vars(settings)
# # print(image_input_size)


# In[80]:


PIL.Image.MAX_IMAGE_PIXELS = 933120000
# IMAGE_INPUT_SIZE = (2048, 2048)

IMAGE_INPUT_SIZE = image_input_size


# In[81]:


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# In[82]:


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


# In[83]:


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


# In[84]:


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


# In[85]:


create_dir_if_not_exist(LOCAL_DATASET_DIR)


# In[86]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')


# In[87]:


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





# In[88]:


dict_id_to_label = {i: label for i, label in enumerate(all_labels)}
dict_label_to_id = {label: i for i, label in enumerate(all_labels)}


# In[89]:


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


# In[90]:


train_image_path_and_label = tran_csv_to_img_path_and_label(train_csv, LOCAL_DATASET_DIR, TRAIN_IMAGE_FOLDER, dict_label_to_id)
test_image_path_and_label = tran_csv_to_img_path_and_label(test_csv, LOCAL_DATASET_DIR, EVAL_IMAGE_FOLDER, dict_label_to_id)

# # Random split
# # train_set, valid_set = train_test_split(train_image_path_and_label, test_size=0.2, random_state=42)
# train_set, valid_set = train_test_split(train_image_path_and_label, test_size=0.2, random_state=42, 
#                                         stratify=[x[1] for x in train_image_path_and_label])
train_set = train_image_path_and_label
valid_set = test_image_path_and_label


print("train set size:", len(train_set))    
print("valid set size:", len(valid_set))
# print("test set size:", len(test_set))

path_list, labels = zip(*train_set)
print("train set category distribution: \n\t", Counter(labels))

path_list, labels = zip(*valid_set)
print("train set category distribution: \n\t", Counter(labels))


# In[91]:


def show_img_by_path(img_path):
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    print(img.size, type(img))
    img = transforms.Resize(IMAGE_INPUT_SIZE)(img)
    print(img.size, type(img))
    img = transforms.ToTensor()(img)
    print(img.shape)
    
# show_img_by_path(train_set[0][0])


# In[92]:


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


# In[93]:


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


# In[94]:


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# In[95]:


#show image grid
def show_image_grid(dataloader, num_of_images=16):
    imgs, labels = next(iter(dataloader))
    image_grid = torchvision.utils.make_grid(imgs[:num_of_images], nrow=8)
    plt.imshow(image_grid.permute(1, 2, 0).numpy())
    # plt.show()
    print("labels:", labels[:num_of_images], [dict_id_to_label[label.item()] for label in labels[:num_of_images]])

# show_image_grid(train_dataloader)


# In[96]:


# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_v2_l', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=True)

# print(dir(efficientnet.modules))
# print(efficientnet.classifier.fc.in_features)


# In[97]:


# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)

# if EXPERIMENT_NAME == "efficientnet_b0":
#     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# elif EXPERIMENT_NAME == "efficientnet_b4":
#     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
# elif EXPERIMENT_NAME == "efficientnet_widese_b0":
#     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
# elif EXPERIMENT_NAME == "efficientnet_widese_b4":
#     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b4', pretrained=True)
# # elif EXPERIMENT_NAME == "efficientnet_v2_l":
# #     efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_v2_l', pretrained=True)

# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')


unet_model = Unet(3, 3).to(device)
bridging_model = BridgingModel().to(device)

X = torch.zeros((1, 3, IMAGE_INPUT_SIZE[0], IMAGE_INPUT_SIZE[1])).to(device)
unet_out, hidden_features = unet_model(X)
(x1, x2, x3, x4, x5) = hidden_features
fs = bridging_model(x1, x2, x3, x4, x5)
B, C, H, W = fs[0].shape
vs = [x.permute(0,2,3,1).reshape(B, H*W, C) for x in fs]

x5 = hidden_features[4].permute(0,2,3,1).reshape(B, H*W, C)
x5_data_dim = x5.shape[1:] 
print(x5_data_dim)

# feature_transformer_model = FeatureTransformer().to(device)
feature_transformer_model = FeatureTransformer(data_dim=x5_data_dim, hidden_d=512, n_heads=16, out_d=5).to(device)


model_list = [unet_model, bridging_model, feature_transformer_model]


# efficientnet.eval().to(device)
# vars(efficientnet)


# In[ ]:


# model_raw = efficientnet
# classifier_in_feature_size = model_raw.classifier.fc.in_features
# classifier_head = torch.nn.Sequential(
#     torch.nn.AdaptiveAvgPool2d(output_size=1),
#     torch.nn.Flatten(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=classifier_in_feature_size, out_features=256, bias=True),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(in_features=256, out_features=5, bias=True),
# )
# model_raw.classifier = classifier_head
# model_raw = model_raw.to(device)


# feature_extractor_params = []
# for name, params in model_raw.named_parameters():
#     if(name[:10] == "classifier"):
#         continue
#     feature_extractor_params.append(params)

# classifier_head_params = []
# for name, params in classifier_head.named_parameters():
#     classifier_head_params.append(params)


# In[ ]:


# print("feature_extractor_params:", len(feature_extractor_params))
# print(classifier_head_params)


# In[ ]:


# summary(model_raw, (3, ) + IMAGE_INPUT_SIZE, device=device.type)


# In[ ]:



criteria = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model_raw.classifier.parameters(), lr=lr, weight_decay=weight_decay)

optimizer = optim.Adam([{"params":unet_model.parameters(), "lr":lr}, 
                        {"params":bridging_model.parameters(), "lr":lr}, 
                        {"params":feature_transformer_model.parameters(), "lr":lr}], 
                       lr=lr, weight_decay=weight_decay)


# In[ ]:




def get_category_accuracy(y_gt: np.array, y_pred: np.array, n_category):
    # category_accuracy = np.zeros(n_category)

    assert(len(y_gt) == len(y_pred))
    assert(len(y_gt.shape) == 1 and len(y_pred.shape) == 1)

    cat_mask_2d = (y_gt == np.arange(n_category).reshape(-1, 1))
    category_accuracy = ((y_gt == y_pred) * cat_mask_2d).mean(axis=1)

    return category_accuracy

# a = np.random.randint(0, 5, 10)
# b = np.random.randint(0, 5, 10)
# print("a:", a)
# print("b:", b)

# print("get_category_accuracy:", get_category_accuracy(y_gt=a, y_pred=b, n_category=5))


# In[ ]:


def evaluation(model_list, valid_dataloader, criteria, device):
        for model in model_list:
            model.eval()

        unet_model, bridging_model, feature_transformer_model = model_list

        valid_loss = 0.0
        valid_corrects = 0

        y_gt = []
        y_pred = []

        for imgs, labels in tqdm(valid_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                y_pred_ep, out, unet_out = get_unet_transformer_model_output(imgs, unet_model, bridging_model, feature_transformer_model)
                outputs = y_pred_ep
                # outputs = model(imgs)
                _, preds = torch.max(outputs, -1)
                loss = criteria(outputs, labels)

            valid_loss += loss.item() * imgs.size(0)
            valid_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()

            y_gt.extend(labels.data.cpu().numpy().reshape(-1))
            y_pred.extend(preds.cpu().numpy().reshape(-1))
            
        confusion_matrix_result = confusion_matrix(y_gt, y_pred)
        print("confusion_matrix_result:\n" + str(confusion_matrix_result))
        logging.info("confusion_matrix_result:\n" + str(confusion_matrix_result))

        category_accuracy = get_category_accuracy(y_gt=np.array(y_gt), y_pred=np.array(y_pred), n_category=5)
        print("get_category_accuracy:" + str(category_accuracy))
        logging.info("get_category_accuracy:" + str(category_accuracy))
        
        valid_loss = valid_loss / len(valid_dataloader.dataset)
        valid_acc = valid_corrects / len(valid_dataloader.dataset)

        valid_balanced_acc = balanced_accuracy_score(y_gt, y_pred)

        return valid_loss, valid_acc, valid_balanced_acc


# In[ ]:


def train(model_list, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, eval_patience, device):
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

        for model in model_list:
            model.train()

        unet_model, bridging_model, feature_transformer_model = model_list

        train_loss = 0.0
        train_corrects = 0
        
        y_gt = []
        y_pred = []

        for imgs, labels in tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            y_pred_ep, out, unet_out = get_unet_transformer_model_output(imgs, unet_model, bridging_model, feature_transformer_model)
            outputs = y_pred_ep
            # outputs = model(imgs)
            _, preds = torch.max(outputs, -1)
            
            loss_cat = criteria(outputs, labels)
            loss_img = (unet_out - imgs).pow(2).mean()
            loss = loss_cat + loss_img

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
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
        logging.info(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        valid_loss, valid_acc, valid_balanced_acc = evaluation(model_list, valid_dataloader, criteria, device)

        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)
        valid_balanced_acc_list.append(valid_balanced_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_valid_loss = copy.deepcopy(model_list)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model_valid_acc = copy.deepcopy(model_list)
        # else:
        #     counter_eval_not_improve += 1

        if valid_balanced_acc > best_valid_balanced_acc:
            best_valid_balanced_acc = valid_balanced_acc
            best_model_valid_balanced_acc = copy.deepcopy(model_list)
        else:
            counter_eval_not_improve += 1


        print(f'Valid loss: {valid_loss:.4f} Acc: {valid_acc:.4f} Balanced Acc: {valid_balanced_acc:.4f}')
        logging.info(f'Valid loss: {valid_loss:.4f} Acc: {valid_acc:.4f} Balanced Acc: {valid_balanced_acc:.4f}')
        elapsed_time = time.time() - start_time
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
        logging.info(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

        if counter_eval_not_improve >= eval_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            logging.info(f'Early stopping at epoch {epoch + 1}')
            break
        else:
            counter_eval_not_improve = 0

    return model_list, best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc,            train_loss_list, train_acc_list, train_balanced_acc_list,            valid_loss_list, valid_acc_list, valid_balanced_acc_list,            best_valid_loss, best_valid_acc


# In[ ]:


def store_result(best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc,                  train_loss_list, train_acc_list, train_balanced_acc_list,                  valid_loss_list, valid_acc_list, valid_balanced_acc_list):
    create_dir_if_not_exist(MODEL_SAVE_DIR)
    create_dir_if_not_exist(RESULT_DIR)

    model_name_list = ["unet_model", "bridging_model", "feature_transformer_model"]
    for (model, model_name) in zip(best_model_valid_acc, model_name_list):
        torch.save(model.state_dict(), Path(MODEL_SAVE_DIR) / f"{model_name}_best_model_valid_acc.pth")

    for (model, model_name) in zip(best_model_valid_loss, model_name_list):
        torch.save(model.state_dict(), Path(MODEL_SAVE_DIR) / f"{model_name}_best_model_valid_loss.pth")

    for (model, model_name) in zip(best_model_valid_balanced_acc, model_name_list):
        torch.save(model.state_dict(), Path(MODEL_SAVE_DIR) / f"{model_name}_best_model_valid_balanced_acc.pth")

    # torch.save(best_model_valid_acc.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_acc.pth")
    # torch.save(best_model_valid_loss.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_loss.pth")
    # torch.save(best_model_valid_balanced_acc.state_dict(), Path(MODEL_SAVE_DIR) / "best_model_valid_balanced_acc.pth")

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


# In[ ]:


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


# In[ ]:


model_list_trained, best_model_valid_acc, best_model_valid_loss, best_model_valid_balanced_acc, train_loss_list, train_acc_list, train_balanced_acc_list, valid_loss_list, valid_acc_list, valid_balanced_acc_list, best_valid_loss, best_valid_acc = train(model_list, train_dataloader, valid_dataloader, optimizer, criteria, num_epochs, eval_patience, device)

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

