#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)


# In[17]:




# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.paths = self.data_df['filename'].to_numpy()
        self.labels = self.init_labels()

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
#         path = self.data_df.loc[index, "filename"]
#         label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label
    
    def __len__(self):
        return len(self.data_df)

    def init_labels(self):
        label_table = {}
        labels = []
        for label in self.data_df['label']:
            if label not in label_table:
                label_table[label] = len(label_table)
            labels.append(label_table[label])

        self.label_table = label_table
        return labels
    
        


# In[9]:


# csv_path = '../hw4_data/train.csv'
# data_dir = '../hw4_data/train'
# train_dataset = MiniDataset(csv_path, data_dir)
# train_dataset.labels

# val_csv_path = '../hw4_data/val.csv'
# val_data_dir = '../hw4_data/val'
# val_dataset = MiniDataset(val_csv_path, val_data_dir)


# In[3]:


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


# In[ ]:


# csv_path = '../hw4_data/train.csv'
# data_dir = '../hw4_data/train'
# train_dataset = MiniDataset(csv_path, data_dir)


# In[4]:


# episode_file_path = '../hw4_data/val_testcase.csv'
# s = GeneratorSampler(episode_file_path)
# s.sampled_sequence
# len(np.unique(s.sampled_sequence))


# In[6]:





# In[ ]:




