# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:39:41 2020

train a graph embedding using metapath2vector
https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf

@author: Xiaocheng
"""
from collections import defaultdict
import networkx as nx
from IIOUtils import *
from Logger import DefaultLogger
from HeterogeneousGraph import HeterogeneousGraph
from MetaPathSchema import MetaPathSchema
from SkipGramDataSet import SkipGramDataSet
from SkipGramDataProvider import SkipGramDataProvider
from RandomWalk import RandomWalk
from SkipGramModel import SkipGramModel
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import argparse

class MetaPath2VecTrainer:
    def __init__(self, 
                 walks, 
                 output_file_path, 
                 emb_dimension=100, 
                 batch_size=32, 
                 num_epoch=3, 
                 num_batch=100000,
                 negative_sample_size=5, 
                 window_size=5,
                 initial_lr=0.001, 
                 logger=DefaultLogger(True)):
        self.logger = logger
        data_provider = SkipGramDataProvider(walks=walks, negative_sample_size=negative_sample_size, window_size=window_size, logger=logger)
        self.dataset = SkipGramDataSet(data_provider)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.dataset.collate)
        self.output_file_path = output_file_path
        self.emb_size = len(self.dataset.data_provider.id2node)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.initial_lr = initial_lr
        self.model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info('device={}'.format(self.device))
        if self.use_cuda:
            self.model.cuda()
        self.__log_config__()

    def __log_config__(self):
        self.logger.info('start show training configuration')
        self.logger.info('emb_size={}'.format(self.emb_size))
        self.logger.info('emb_dimension={}'.format(self.emb_dimension))
        self.logger.info('batch_size={}'.format(self.batch_size))
        self.logger.info('num_epoch={}'.format(self.num_epoch))
        self.logger.info('num_batch={}'.format(self.num_batch))
        self.logger.info('initial_lr={}'.format(self.initial_lr))
        self.logger.info('output_file_path={}'.format(self.output_file_path))   
        self.logger.info('end show training configuration')        

    def train(self):
        self.logger.info('start train')
        for epoch in range(self.num_epoch):
            self.logger.info("epoch={} started.".format(epoch + 1))            
            optimizer = optim.SparseAdam(self.model.parameters(), lr=self.initial_lr)           
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_batch)
            total_loss = 0.0
            bucket_size = 1
            bucket = bucket_size
            for i, sample_batched in enumerate(self.dataloader):
                if i >= self.num_batch:
                    break
                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    cur_loss = loss.item()
                    total_loss += cur_loss
                    if 1000 * (i + 1) / self.num_batch >= bucket:
                        self.logger.info("batch: {}/{}({:.2%}), samples: average loss={}; current loss={}".format(i + 1, self.num_batch, (i + 1) / self.num_batch, total_loss / (i + 1), cur_loss))
                        bucket += bucket_size
                    self.logger.debug("batch: {}/{}({:.2%}), samples: average loss={}; current loss={}".format(i + 1, self.num_batch, (i + 1) / self.num_batch, total_loss / (i + 1), cur_loss))
        self.logger.info('save embedding to path: {}'.format(self.output_file_path))
        self.model.save_embedding(self.dataset.data_provider.id2node, self.output_file_path)
        self.logger.info('end train')

# python MetaPath2VectorTrain.py -i D:\data\net_aminer_test\random_walks.txt -o D:\data\net_aminer_test\node_embedding.txt -num_batch 5 -debug false
# python MetaPath2VectorTrain.py -i D:\data\net_aminer\random_walks.txt -o D:\data\net_aminer\node_embedding.txt -num_batch 1000000 -num_epoch 5 -debug false
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="walk file path", type=str)
    parser.add_argument("-o", help="output model file path", type=str)
    parser.add_argument("-emb_dimension", help="embedding dimension", type=int, default=128)
    parser.add_argument("-batch_size", help="training batch size", type=int, default=32)
    parser.add_argument("-num_epoch", help="training epoch number", type=int, default=3)
    parser.add_argument("-num_batch", help="training epoch number", type=int, default=10000)
    parser.add_argument("-negative_sample_size", help="negative sample size", type=int, default=5)
    parser.add_argument("-window_size", help="skip gram context window szie", type=int, default=5)
    parser.add_argument("-initial_lr", help="learning rate", type=float, default=0.001)    
    parser.add_argument("-debug", help="debug flag", type=str, default="false")    
    args = parser.parse_args()
    walks = [json.loads(line) for line in get_data_row(args.i)]    
    trainer = MetaPath2VecTrainer(
        walks=walks,
        output_file_path=args.o,
        emb_dimension=args.emb_dimension,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        num_batch=args.num_batch,
        negative_sample_size=args.negative_sample_size,
        window_size=args.window_size,
        initial_lr=args.initial_lr,
        logger=DefaultLogger(args.debug.lower() == 'true')
        )
    trainer.train()