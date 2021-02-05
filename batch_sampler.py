# coding=utf-8
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class BatchSampler(object):
    def __init__(self, mode, labels, classes_per_iter, sample_per_class, episodes):
        super(BatchSampler, self).__init__()
        assert mode in ["train", "val"]
        self.mode = mode
        self.labels = labels
        # batch size = classes_per_iter * sample_per_class
        self.classes_per_iter = classes_per_iter  # N-way
        self.sample_per_class = sample_per_class  # N-support (N-shot) + N-query
        self.episodes = episodes  # number of episodes per epoch = dataset size / batch size

        self.classes = np.arange(64) if mode == "train" else np.arange(16)
        self.count = 600  # number of samples for each classes in the dataset
        self.classes = torch.LongTensor(self.classes)

        self.indexes = [[i for i in range(600*c, 600*(c+1))] for c in self.classes]
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = ([600 for i in range(len(self.classes))])
        
        # init sampled sequence
        self.sampled_sequence = self._init_sampled_sequence()

    def __iter__(self):
        return iter(self.sampled_sequence)

    def __len__(self):
        return len(self.sampled_sequence)

    # There may be duplicated samples in one epoch when training    
    def _init_sampled_sequence(self):
        seqs = []
        
        spc = self.sample_per_class
        cpi = self.classes_per_iter
        batch_size = spc * cpi

        for it in range(self.episodes):
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            seqs.append(batch)
        
        return torch.cat(seqs).tolist()

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)


# class BatchSampler(object):
#     def __init__(self, labels, classes_per_it, num_samples, iterations):
#         super(BatchSampler, self).__init__()
#         self.labels = labels
#         self.classes_per_it = classes_per_it
#         self.sample_per_class = num_samples
#         self.iterations = iterations

#         self.classes, self.counts = np.unique(self.labels, return_counts=True)
#         self.classes = torch.LongTensor(self.classes)

#         self.idxs = range(len(self.labels))
#         self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
#         self.indexes = torch.Tensor(self.indexes)
#         self.numel_per_class = torch.zeros_like(self.classes)
#         for idx, label in enumerate(self.labels):
#             label_idx = np.argwhere(self.classes == label).item()
#             self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
#             self.numel_per_class[label_idx] += 1

#     def __iter__(self):
#         spc = self.sample_per_class
#         cpi = self.classes_per_it

#         for it in range(self.iterations):
#             batch_size = spc * cpi
#             batch = torch.LongTensor(batch_size)
#             c_idxs = torch.randperm(len(self.classes))[:cpi]
#             for i, c in enumerate(self.classes[c_idxs]):
#                 s = slice(i * spc, (i + 1) * spc)
#                 label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
#                 sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
#                 batch[s] = self.indexes[label_idx][sample_idxs]
#             batch = batch[torch.randperm(len(batch))]
#             yield batch

#     def __len__(self):
#         return self.iterations
