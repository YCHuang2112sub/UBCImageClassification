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

import argparse
import re


parser = argparse.ArgumentParser(description='Downsample UCB Image Dataset')
parser.add_argument('--source_dataset_dir', type=str, default='/projectnb/cs640grp/materials/UBC-OCEAN_CS640', help='path to dataset')
parser.add_argument('--target_dataset_dir', type=str, default='./dataset', help='path to local dataset')
parser.add_argument('--model_dir', type=str, default='./model', help='path to tained model')

parser.add_argument('--source_train_image_folder', type=str, default='train_images_compressed_80', help='training image folder name')
parser.add_argument('--source_test_image_folder', type=str, default='test_images_compressed_80', help='training image folder name')
parser.add_argument('--image_downsampled_size', type=str, default='(256, 256)', help='input image size')

setting = None
try:
    __IPYTHON__
    _in_ipython_session = True
    settings = parser.parse_args("")
except NameError:
    _in_ipython_session = False
    settings = parser.parse_args()

image_downsampled_size = eval(settings.image_downsampled_size)
assert isinstance(image_downsampled_size, tuple) and len(image_downsampled_size) == 2, "image_input_size must be a tuple of length 2"

print("settings:", vars(settings))
# # save settings in to a log file
# with open(Path(settings.model_dir, settings.experiment_name, 'settings.txt'), 'w') as f:
#     print(vars(settings), file=f)

PIL.Image.MAX_IMAGE_PIXELS = 933120000

# IMAGE_INPUT_SIZE = (2048, 2048)

# IMAGE_DOWN_SAMPLED_SIZE = (256, 256)
# IMAGE_DOWN_SAMPLED_SIZE = (512, 512)
# IMAGE_DOWN_SAMPLED_SIZE = (1024, 1024)
# IMAGE_DOWN_SAMPLED_SIZE = (2048, 2048)
IMAGE_DOWN_SAMPLED_SIZE = image_downsampled_size

# DATASET_PATH = "/projectnb/cs640grp/materials/UBC-OCEAN_CS640"
# TRAIN_IMAGE_FOLDER = "train_images_compressed_80"
# TEST_IMAGE_FOLDER = "test_images_compressed_80"
DATASET_PATH = settings.source_dataset_dir
TRAIN_IMAGE_FOLDER = settings.source_train_image_folder
TEST_IMAGE_FOLDER = settings.source_test_image_folder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f'Using {device} for inference')

train_csv = pd.read_csv(Path(DATASET_PATH) / 'train.csv').to_numpy()
test_csv = pd.read_csv(Path(DATASET_PATH) / 'test.csv').to_numpy()

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# TARGET_DATASET_PATH = "./dataset"
TARGET_DATASET_PATH = settings.target_dataset_dir
TARGET_IMAGE_FOLDER = f"img_size_{IMAGE_DOWN_SAMPLED_SIZE[0]}x{IMAGE_DOWN_SAMPLED_SIZE[1]}"
print(f"target image folder: {TARGET_IMAGE_FOLDER}")

create_dir_if_not_exist(TARGET_DATASET_PATH)


def load_image(img_name, image_source_dir_path):
    img_path = image_source_dir_path / img_name
    
    # check image is exist
    if not img_path.exists():
        return None

    #load image
    img = Image.open(img_path)
    return img

def transform_image(img, target_size):
    return transforms.Resize(target_size)(img)

def save_image(img, img_name, image_save_dir_path):
    img_save_path = image_save_dir_path / img_name
    img.save(img_save_path)



## resize train images
source_dataset_path = DATASET_PATH
source_image_folder = TRAIN_IMAGE_FOLDER
target_dataset_path = TARGET_DATASET_PATH
target_image_folder = TARGET_IMAGE_FOLDER
x_csv = train_csv

image_source_dir_path = Path(source_dataset_path) / source_image_folder
print(f"source dir: {image_source_dir_path}")

image_save_dir_path = Path(target_dataset_path) / target_image_folder
create_dir_if_not_exist(image_save_dir_path)
print(f"target dir: {image_save_dir_path}")

start_time = time.time()

for i in tqdm(range(len(x_csv))):
    img_name = str(x_csv[i][0]) + ".jpg"
    
    #load image
    source_img = load_image(img_name, image_source_dir_path)
    if(source_img == None):
        continue

    #transform image
    img_down_sampled = transform_image(source_img, IMAGE_DOWN_SAMPLED_SIZE)

    #save image
    save_image(img_down_sampled, img_name, image_save_dir_path)

    if(i % 50 == 0):
        print(f"{i+1} images has been processed")

        elapsed_time = time.time() - start_time
        # forat into hh:mm:ss
        print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    
elapsed_time = time.time() - start_time
# forat into hh:mm:ss
print(f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

