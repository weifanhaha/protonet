#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.nn.modules import Module

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from protonet import ProtoNet
from mini_dataset import MiniDataset
import csv
import random
import numpy as np
import pandas as pd

from PIL import Image


def filenameToPILImage(x): return Image.open(x)


# In[2]:


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# In[6]:


class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)


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


def expand_vecs(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return x, y


def euclidean_dist(x, y):
    x, y = expand_vecs(x, y)

    return torch.pow(x - y, 2).sum(2)


# In[4]:


def predict(model, data_loader, N_way, N_shot):
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        prediction_results = []
        for i, (data, target) in enumerate(data_loader):
            print("\r{} / {}".format(i, len(data_loader)), end=' ', flush=True)

            # split data into support and query data
            support_input = data[:N_way * N_shot, :, :, :]
            query_input = data[N_way * N_shot:, :, :, :]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * N_shot]: i for i in range(N_way)}
            query_label = torch.cuda.LongTensor(
                [label_encoder[class_name] for class_name in target[N_way * N_shot:]])

            with torch.no_grad():
                supports = model(support_input)
                querys = model(query_input)

            dists = euclidean_dist(querys, supports)
            log_p_y = F.log_softmax(-dists, dim=1).view(N_way, N_query, -1)

            target_inds = torch.arange(0, N_way)
            target_inds = target_inds.view(N_way, 1, 1)
            target_inds = target_inds.expand(N_way, N_query, 1).long()
            loss_val = - \
                log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            _, y_hat = log_p_y.max(2)
            prediction_results.append(y_hat.flatten())

    return prediction_results


# In[5]:


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int,
                        help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int,
                        help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int,
                        help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str,
                        help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()


# In[ ]:


if __name__ == '__main__':
    args = parse_args()

    N_way = args.N_way
    N_shot = args.N_shot
    N_query = args.N_query
    model_path = args.load
    test_csv = args.test_csv
    test_data_dir = args.test_data_dir
    testcase_csv = args.testcase_csv
    output_csv = args.output_csv

    test_dataset = MiniDataset(test_csv, test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=N_way * (N_query + N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(testcase_csv))

    model = ProtoNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict((torch.load(
        model_path, map_location=device)))

    prediction_results = predict(model, test_loader, N_way, N_shot)

    row_names = ["query{}".format(i) for i in range(1, 76)]
    row_names = ["episode_id"] + row_names

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_names)

        for i, pred in enumerate(prediction_results):
            row = [i] + list(pred.numpy())
            writer.writerow(row)
