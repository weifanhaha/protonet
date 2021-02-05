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
from halliculator import Halliculator
from loss_utils import euclidean_dist


# In[26]:


N_query = 3
val_N_query = 15
N_shot = 1
# training N_way can > validation N_way, 30 for paper
N_way = 5
val_N_way = 5
sample_M = 1

batch_size = N_way * (N_query + N_shot)
val_batch_size = val_N_way * (val_N_query + N_shot)

sample_per_class = N_query + N_shot
val_sample_per_class = val_N_query + N_shot

# episodes can be larger than int(len(train_dataset) /  batch_size)
# => it depends on how long you want to train
episodes = 800
num_epochs = 120
# total episodes = epoch * episodes

lr = 0.001
lr_step = (num_epochs) // 5
lr_gamma = 0.5

# paths
csv_path = '../hw4_data/train.csv'
data_dir = '../hw4_data/train'
val_csv_path = '../hw4_data/val.csv'
val_data_dir = '../hw4_data/val'
model_path = './models/best_model_M1_q3.pth'
generator_path = './models/best_generator_M1_q3.pth'


# In[27]:


train_dataset = MiniDataset(csv_path, data_dir)
train_sampler = BatchSampler('train', train_dataset.labels, N_way, sample_per_class, episodes)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size,
    sampler = train_sampler
)


# In[28]:


testcase_csv = '../hw4_data/val_testcase.csv'
test_dataset = MiniDataset(val_csv_path, val_data_dir)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

val_dataloader = DataLoader(
    test_dataset, batch_size=val_batch_size,
    num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
    sampler=GeneratorSampler(testcase_csv))

# replace with val data loader ta give
# val_dataset = MiniDataset(val_csv_path, val_data_dir)
# val_sampler = BatchSampler('val', val_dataset.labels, val_N_way, val_sample_per_class, episodes)
# val_dataloader = DataLoader(
#     val_dataset, batch_size=val_batch_size,
#     sampler = val_sampler
# )


# In[29]:


import torch
import torch.nn as nn

# define model
class Halliculator(nn.Module):
    def __init__(self, featdim, device, innerdim=1600):
        super(Halliculator,self).__init__()
        self.featdim = featdim
        self.device = device
        self.innerdim = innerdim
        
        self.fc1 = nn.Linear(featdim, innerdim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(innerdim, innerdim)
        self.fc3 = nn.Linear(innerdim, featdim)
        
#         self.fc1.weight.data.copy_(torch.eye(innerdim))
#         self.fc2.weight.data.copy_(torch.eye(innerdim))
#         self.fc3.weight.data.copy_(torch.eye(innerdim))

        
    def forward(self, x, add_noise=True):
        noise = torch.empty(x.shape).normal_(mean=0,std=1).to(self.device)
        x = x + noise
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


# In[30]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ProtoNet()
model = model.to(device)

generator = Halliculator(featdim=1600, device=device)
generator = generator.to(device)


# In[31]:


# x, y = next(iter(train_dataloader))
# x = x.to(device)
# y = y.to(device)
# out = model(x)
# generator(out)
# ps, qs = get_proto_query(out.cpu(), y.cpu(), n_aug=1, n_support=1)


# In[32]:


# ps.shape


# In[33]:


# ps


# In[34]:


parameters = list(model.parameters()) + list(generator.parameters())
optim = torch.optim.Adam(params=parameters, lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=lr_gamma,
                                           step_size=lr_step)


# In[35]:


def get_proto_query(input, target, n_aug, n_support):
    def get_supp_idxs(c):
        return target.eq(c).nonzero()[:n_support].squeeze(1)

    def get_query_idxs(c):
        return target.eq(c).nonzero()[n_support:].squeeze(1)

    classes = torch.unique(target)
    n_classes = len(classes)

    support_idxs = torch.stack(list(map(get_supp_idxs, classes)))
    support_samples = input[support_idxs]

    query_idxs = torch.stack(list(map(get_query_idxs, classes)))
    query_samples = input[query_idxs]
    
    if n_aug <= 0:
        prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
        return prototypes, query_samples.view(-1, 1600)
    
    # halliculate
    supports = []
    queries = []
    for c in range(n_classes):
        support = []
        query = []
        
        # sample n_aug queries from query_samples
        c_query_idxs = query_idxs[c]
        while len(c_query_idxs) < n_aug:
            c_query_idxs = torch.cat([c_query_idxs, c_query_idxs])
            
        sampled_idxs = torch.randperm(len(c_query_idxs))[:n_aug]
        sampled_query_idxs = c_query_idxs[sampled_idxs]

        for i in range(n_aug):
            # halliculate support
            gen = generator(support_samples[c][0].to(device)).cpu()
            support.append(gen)
            
            # halliculate query
            gen_query = generator(input[sampled_query_idxs[i]].to(device)).cpu()
            query.append(gen_query)

        support.append(support_samples[c][0])
        supports.append(support)
        
        for q in query_samples[c]:
            query.append(q)
        queries.append(torch.stack(query))

    # get prototype and query_samples
    prototype = torch.stack([torch.stack(supports[i]).mean(0) for i in range(len(supports))])
    query_samples = torch.stack(queries).view(-1, 1600)
    return prototype, query_samples
#     return supports, queries


# In[36]:


# prototype, query_samples, supports = get_proto_query(model_output.cpu(), y.cpu(), n_aug, n_support)
# query_samples.shape


# In[44]:


def cal_loss(query_samples, prototypes, n_classes=N_way, n_aug=0):
#     n_samples = N_query + n_aug
    n_samples = query_samples.shape[0] // n_classes
    
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_samples, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_samples, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val,  acc_val


# In[45]:


train_loss = []
train_acc = []
val_loss = []
val_acc = []
best_acc = 0
best_model = None
best_generator = None

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
        prototype, query_samples = get_proto_query(model_output.cpu(), y.cpu(), n_aug=sample_M, n_support=N_shot)
#         loss, acc = loss_fn(query_samples, prototype, n_aug=sample_M)
        loss, acc  = cal_loss(query_samples, prototype, n_aug=sample_M)

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
            prototype, query_samples = get_proto_query(model_output.cpu(), y.cpu(), n_aug=0, n_support=N_shot)
            loss, acc = cal_loss(query_samples, prototype, n_classes=val_N_way, n_aug=0)            

        val_loss.append(loss.item())
        val_acc.append(acc.item())

        postfix_dict = {
            "val_loss": loss.item(),
            "val_acc": acc.item()
        }
        val_trange.set_postfix(**postfix_dict)
        
    avg_loss = np.mean(val_loss[-600:])
    avg_acc = np.mean(val_acc[-600:])
    postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
        best_acc)
    print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
    
    
    if avg_acc >= best_acc:
        best_acc = avg_acc
        best_model = copy.deepcopy(model.state_dict())
        best_generator = copy.deepcopy(generator.state_dict())
    if epoch % 5 == 0 and best_model != None:
        torch.save(best_model, model_path)
        torch.save(best_generator, generator_path)
        
        print("model saved! - {}".format(model_path))
        
     # init sampler
    train_sampler = BatchSampler('train', train_dataset.labels, N_way, sample_per_class, episodes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler = train_sampler
    )
#     val_sampler = BatchSampler('val', val_dataset.labels, val_N_way, val_sample_per_class, episodes)
#     val_dataloader = DataLoader(
#         val_dataset, batch_size=val_batch_size,
#         sampler = val_sampler
#     )


# In[43]:


if best_model is not None:
    torch.save(best_model, model_path)
    torch.save(best_generator, generator_path)


# In[ ]:




