# -*- coding: utf-8 -*-
"""
Random walk torch dataset

Created on Wed May  6 21:39:55 2020

@author: Xiaocheng
"""

import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset
from Logger import DefaultLogger
import random

class RandomWalkDataset(Dataset):    
    # the freq ratio, please refer to http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    RATIO = 3.0 / 4.0 
    def __init__(self, walks, negative_sample_size=5, window_size= , ratio=RATIO, loggger=DefaultLogger(True)):
        """
        the inpt data is a list of walks and each element in the walk is the id of the node        
        window_size is the window size for skip-gram model
        refer https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        """
        self.window_size = window_size
        self.negative_sample_size = negative_sample_size
        self.ratio = ratio
        self.logger = logger        
        np.random.seed(0)
        random.seed(0)
        self.logger.info('initalize RandonDataDataSet with configuration: {}'.formmat(self.__get_config__()))
        self.__construct_samples__(walks)
    
    def __get_config__(self):
        return 'window_size={};negative_sample_size={};freq_ratio={}'.format(self.window_size, self.negative_sample_size, self.ratio)
    
    def __construct_samples__(self, walks):
        """
        1 get node frequency
        2 construct skip-gram model training samples
        """
        self.logger.info('start to construct training samples')
        self.logger.info('read {} walks'.format(len(self.walks)))
        freq = collections.Counter()
        for walk in walks:
            for node in walk:
                freq[node] += 1
        self.logger.info('read {} nodes'.format(len(freq)))        

        self.node_index = {}      
        self.index_node = []
        probs = []
        for node, count in freq.items():
            self.nodex_index[node] = len(probs)
            self.index_node.append(node)
            probs.append(count)
        self.probs = np.array(probs) ** RATIO
        self.probs = self.probs / np.sum(self.probs)
        self.logger.debug('negative sample probs: {}'.format(self.probs))
        
        self.samples = []
        for walk in walks:            
            for i, node in enumerate(walk):
                left = max(0, i - self.window_size)
                right = min(len(walk) - 1, i + self.window_size)
                for j in range(left, right + 1):
                    if j != i:
                        self.samples.append((walk[i], node, self.get_negative_samples(node)))
        random.shuffle(self.samples)
        self.logger.info('total seamples: {}'.format(len(self.samples)))
        self.logger.info('end to construct training samples')
        
    def get_negative_samples(self, node):
        """        
        given node, do negative sampling, 
        negative exmample should not be the same with the target node

        """
        negative_samples = set()
        index = self.node_index[node]
        cnt = 0
        while len(negative_samples) < self.negative_sample_size:
            sample_index = np.random.choice(len(self.nodex_index), size=1, replace=False, p=self.probs)
            if sample_index != index:
                negtive_samples.add(sample_index)
            cnt += 1
        #self.logger.debug('samples {} times to generate {} negative samples'.format(cnt, self.negative_sample_size))
        return [self.index_node[i] for i in negative_samples]
               
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)

