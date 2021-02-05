#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn


# In[3]:


# define model
class Halliculator(nn.Module):
    def __init__(self, featdim, device, innerdim=512):
        super(Halliculator,self).__init__()
        self.featdim = featdim
        self.device = device
        self.innerdim = innerdim
        
        self.fc1 = nn.Linear(featdim, innerdim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(innerdim, innerdim)
        self.fc3 = nn.Linear(innerdim, featdim)

    def forward(self, x, add_noise=True):
        noise = torch.empty(self.featdim).normal_(mean=0,std=0.1).to(self.device)
        x = x + noise
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


# In[ ]:


# # sample gaussian loss
# from torch.utils.data import DataLoader, Dataset

# from mini_dataset import MiniDataset
# from batch_sampler import BatchSampler, GeneratorSampler

# csv_path = '../hw4_data/train.csv'
# data_dir = '../hw4_data/train'
# val_csv_path = '../hw4_data/val.csv'
# val_data_dir = '../hw4_data/val'

# N_query =15
# N_shot = 1
# # training N_way can > validation N_way, 30 for paper
# N_way = 5
# val_N_way = 5

# batch_size = N_way * (N_query + N_shot)
# val_batch_size = val_N_way * (N_query + N_shot)
# sample_per_class = N_query + N_shot

# # episodes can be larger than int(len(train_dataset) /  batch_size)
# # => it depends on how long you want to train
# episodes = 500
# num_epochs = 100

# val_dataset = MiniDataset(val_csv_path, val_data_dir)
# val_sampler = BatchSampler('val', val_dataset.labels, val_N_way, sample_per_class, episodes)
# val_dataloader = DataLoader(
#     val_dataset, batch_size=val_batch_size,
#     sampler = val_sampler
# )


# In[26]:


# from protonet import ProtoNet

# model = ProtoNet()
# feature_dim = 1600

# image, label = next(iter(val_dataloader))
# features = model(image)
# noise = torch.empty(feature_dim).normal_(mean=0,std=0.1)

# generator = Halliculator(featdim=feature_dim)
# gen = generator(noise)


# In[ ]:




