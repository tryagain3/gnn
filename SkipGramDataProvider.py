# -*- coding: utf-8 -*-
"""
class used to generate training data using skip gram way

http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/    

Created on Thu May 14 20:38:48 2020

@author: Xiaocheng
"""

import numpy as np
import torch
import collections
import random
from Logger import DefaultLogger
RATIO = 3.0 / 4.0
NEGATIVE_TABLE_SIZE = 1e3

class SkipGramDataProvider:    
    # the freq ratio, please refer to http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/    
    # note: set bigger value for NEGATIVE_TABLE_SIZE, if the given graph have larger number of nodes
    def __init__(self, walks, negative_sample_size=5, window_size=5, ratio=RATIO, negative_table_size=NEGATIVE_TABLE_SIZE, logger=DefaultLogger(True)):
        """
        the inpt data is a list of walks and each element in the walk is the id of the node        
        window_size is the window size for skip-gram model
        refer https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        """
        self.window_size = window_size
        self.negative_sample_size = negative_sample_size
        self.ratio = ratio
        self.logger = logger        
        self.negative_table_size = negative_table_size
        np.random.seed(0)
        random.seed(0)
        self.logger.info('initalize RandonDataDataSet with configuration: {}'.format(self.__get_config__()))
        self.walk_index = 0
        self.walks = walks
        self.__build_probs__(walks)
    
    def __get_config__(self):
        return 'window_size={};negative_sample_size={};freq_ratio={};negative_table_size={}'.format(self.window_size, self.negative_sample_size, self.ratio, self.negative_table_size)
    
    def __build_probs__(self, walks):
        """
        1 get node frequency and build sample probs
        2 generate node2index table
        """
        self.logger.info('start to construct training samples')
        self.logger.info('read {} walks'.format(len(walks)))
        freq = collections.Counter()
        for walk in walks:
            for node in walk:
                freq[node] += 1
        self.logger.info('read {} nodes'.format(len(freq)))        

        self.node2id = {}      
        self.id2node = []
        probs = []
        for node, count in freq.items():
            self.node2id[node] = len(probs)
            self.id2node.append(node)
            probs.append(count)
        self.probs = np.array(probs) ** self.ratio
        self.probs = self.probs / np.sum(self.probs)
        self.logger.debug('negative sample probs (top 10): {}'.format(self.probs[:10]))        
        self.build_neg_table()
        
    def __next__(self):
        return self.__gen_samples__()
    
    def __gen_samples__(self):
        """
        generate training data for the given walk in current index
        1 round index, which will make it generate endless training data
        2 output format list([u, v, [neg_v1, neg_v2, ...., neg_v#negative_sample_size]])
        """
        walk = self.walks[self.walk_index]        
        self.logger.debug('start to generate training data (negative samples) for walk: {}'.format(self.walk_index))
        samples = []                 
        for i, node in enumerate(walk):                
            left = max(0, i - self.window_size)
            right = min(len(walk) - 1, i + self.window_size)
            for j in range(left, right + 1):
                if walk[j] != node:
                        samples.append((self.node2id[walk[j]], self.node2id[node], self.get_negative_samples(self.node2id[node])))
        self.logger.debug('end generate data for walk {}'.format(self.walk_index))
        self.logger.debug('total samples: {}'.format(len(samples)))
        self.walk_index = (self.walk_index + 1) % len(self.walks)
        return samples
    
    """    
    def get_negative_samples(self, node):
        #given node, do negative sampling, 
        #negative exmample should not be the same with the target node
        #this one is too slow, around 0.5 process 1 walk for given Aminet which has 1,693,531 node
        negative_samples = set()
        index = self.node_index[node]
        cnt = 0
        while len(negative_samples) < self.negative_sample_size:
            sample_index = np.random.choice(len(self.node_index), size=self.negative_sample_size, replace=False, p=self.probs)
            for x in sample_index:
                if x != index:
                    negative_samples.add(x)
            cnt += 1
        self.logger.debug('samples {} times to generate {} negative samples'.format(cnt, self.negative_sample_size))
        return [self.index_node[i] for i in negative_samples]
    """
    
    def build_neg_table(self):
        self.logger.info('start generate negative table')
        count = np.round(self.probs * self.negative_table_size)
        self.neg_table = []
        not_sampled_count = 0
        for wid, c in enumerate(count):
            cnt = int(c)
            if cnt == 0:
                not_sampled_count += 1
            self.neg_table.extend([wid] * cnt)
        self.neg_table = np.array(self.neg_table)
        self.neg_pos = 0
        np.random.shuffle(self.neg_table)
        self.logger.info('not sampled count={}'.format(not_sampled_count))
        self.logger.info('end generate negative table')
        
    def get_negative_samples(self, index):
        """        
        given node, do negative sampling, 
        negative exmample should not be the same with the target node
        use negative table to make sample faster
        """
        negative_samples = set()
        cnt = 0
        while len(negative_samples) < self.negative_sample_size:
            if self.neg_table[self.neg_pos] != index:
                negative_samples.add(self.neg_table[self.neg_pos])
            self.neg_pos = (self.neg_pos + 1) % len(self.neg_table)
            cnt += 1
        #self.logger.debug('samples {} times to generate {} negative samples'.format(cnt, self.negative_sample_size))
        return [int(x) for x in negative_samples]

    