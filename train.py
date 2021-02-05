#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import numpy as np
import torch
import os
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import copy

from mini_dataset import MiniDataset
from batch_sampler import BatchSampler, GeneratorSampler
from protonet import ProtoNet
from loss_utils import euclidean_dist, loss_fn


# In[2]:


N_query =15
N_shot = 1
# training N_way can > validation N_way, 30 for paper
N_way = 5
val_N_way = 5

batch_size = N_way * (N_query + N_shot)
val_batch_size = val_N_way * (N_query + N_shot)
sample_per_class = N_query + N_shot

# episodes can be larger than int(len(train_dataset) /  batch_size)
# => it depends on how long you want to train
episodes = 500
num_epochs = 100
# total episodes = epoch * episodes

lr = 0.001
lr_step = (num_epochs) // 5 
lr_gamma = 0.5

# paths
csv_path = '../hw4_data/train.csv'
data_dir = '../hw4_data/train'
val_csv_path = '../hw4_data/val.csv'
val_data_dir = '../hw4_data/val'
model_path = './best_model.pth'


# In[3]:


train_dataset = MiniDataset(csv_path, data_dir)
train_sampler = BatchSampler('train', train_dataset.labels, N_way, sample_per_class, episodes)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size,
    sampler = train_sampler
)


# In[4]:


# testcase_csv = '../hw4_data/val_testcase.csv'
# test_dataset = MiniDataset(test_csv, test_data_dir)

# # fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# random.seed(SEED)
# np.random.seed(SEED)

# def worker_init_fn(worker_id):                                                          
#     np.random.seed(np.random.get_state()[1][0] + worker_id)

# val_loader = DataLoader(
#     test_dataset, batch_size=val_N_way * (N_query + N_shot),
#     num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
#     sampler=GeneratorSampler(testcase_csv))

# replace with val data loader ta give
val_dataset = MiniDataset(val_csv_path, val_data_dir)
val_sampler = BatchSampler('val', val_dataset.labels, val_N_way, sample_per_class, episodes)
val_dataloader = DataLoader(
    val_dataset, batch_size=val_batch_size,
    sampler = val_sampler
)


# In[5]:


# train_dataset = MiniDataset(csv_path, data_dir)
# val_dataset = MiniDataset(val_csv_path, val_data_dir)

# train_sampler = BatchSampler(train_dataset.labels, N_way, sample_per_class, episodes)
# val_sampler = BatchSampler(val_dataset.labels, val_N_way, sample_per_class, episodes)

# train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler)
# val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler)


# In[6]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProtoNet()
model = model.to(device)


# In[7]:


optim = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=lr_gamma,
                                           step_size=lr_step)


# In[8]:


train_loss = []
train_acc = []
val_loss = []
val_acc = []
best_acc = 0
best_model = None

for epoch in range(num_epochs):
    print("Epoch: {}/{}".format(epoch, num_epochs))

    # train
    model.train()
    trange = tqdm(train_dataloader)
    for batch in trange:
        optim.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        model_output = model(x)
        loss, acc = loss_fn(model_output, target=y, n_support=N_shot)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())
        train_acc.append(acc.item())
        postfix_dict = {
            "train_loss": loss.item(),
            "train_acc": acc.item()
        }
        trange.set_postfix(**postfix_dict)
        
    avg_loss = np.mean(train_loss[-episodes:])
    avg_acc = np.mean(train_acc[-episodes:])
    print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
    scheduler.step()

    model.eval()
    val_trange = tqdm(val_dataloader)

    for batch in val_trange:
        with torch.no_grad():
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
        loss, acc = loss_fn(model_output, target=y,
                            n_support=N_shot)
        val_loss.append(loss.item())
        val_acc.append(acc.item())

        postfix_dict = {
            "val_loss": loss.item(),
            "val_acc": acc.item()
        }
        val_trange.set_postfix(**postfix_dict)
        
    avg_loss = np.mean(val_loss[-episodes:])
    avg_acc = np.mean(val_acc[-episodes:])
    postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
        best_acc)
    print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
    
    
    if avg_acc > best_acc:
        best_acc = avg_acc
        best_model = copy.deepcopy(model.state_dict())
    if epoch % 10 == 0 and best_model != None:
        torch.save(best_model, model_path)
        print("model saved!")
        
     # init sampler
    train_sampler = BatchSampler('train', train_dataset.labels, N_way, sample_per_class, episodes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler = train_sampler
    )
    val_sampler = BatchSampler('val', val_dataset.labels, val_N_way, sample_per_class, episodes)
    val_dataloader = DataLoader(
        val_dataset, batch_size=val_batch_size,
        sampler = val_sampler
    )


# In[9]:


if best_model is not None:
    torch.save(best_model, model_path)


# In[ ]:




