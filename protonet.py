#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# In[6]:


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
        


# In[ ]:



# class ProtoNet(nn.Module):
#     '''
#     Model as described in the reference paper,
#     source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
#     '''
#     def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
#         super(ProtoNet, self).__init__()
#         self.encoder = nn.Sequential(
#             conv_block(x_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, hid_dim),
#             conv_block(hid_dim, z_dim),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x.view(x.size(0), -1)

