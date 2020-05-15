# -*- coding: utf-8 -*-
"""
torch dataset to generate data using SkipGram method
https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Created on Wed May  6 21:39:55 2020

@author: Xiaocheng
"""

import numpy as np
import torch
import collections
import random
from torch.utils.data import IterableDataset
from Logger import DefaultLogger
from SkipGramDataProvider import SkipGramDataProvider

class SkipGramDataSet(IterableDataset):    
    def __init__(self, data_provider):
        """
        initialize torch dataset using data_provider which is a SkipGramDataProvider
        """
        self.data_provider = data_provider
               
    def __iter__(self):
        while True:
            yield next(self.data_provider)
    
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)

