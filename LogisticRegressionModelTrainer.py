# -*- coding: utf-8 -*-
"""
use cross entropy loss to train a logistic regression
Created on Tue May 19 20:58:59 2020

@author: Xiaocheng
"""
from LogisticRegressionModel import LogisticRegressionModel
from Logger import DefaultLogger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import argparse

class LogisticRegressionModelTrainer:
    def __init__(self, dataloader, input_size, output_size, batch_size, num_epoch, num_batch, initial_lr, logger=DefaultLogger(True)):
        self.dataloader = dataloader
        self.logger = logger
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.initial_lr = initial_lr
        self.input_size = input_size
        self.output_size = output_size
        self.model = LogisticRegressionModel(self.input_size, self.output_size)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.logger.info('device={}'.format(self.device))
        if self.use_cuda:
            self.model.cuda()
        self.__log_config__()

    def __log_config__(self):
        self.logger.info('start show training configuration')
        self.logger.info('batch_size={}'.format(self.batch_size))
        self.logger.info('num_epoch={}'.format(self.num_epoch))
        self.logger.info('num_batch={}'.format(self.num_batch))
        self.logger.info('initial_lr={}'.format(self.initial_lr))
        self.logger.info('input_size={}'.format(self.input_size))   
        self.logger.info('output_size={}'.format(self.output_size))  
        self.logger.info('end show training configuration')        

    def train(self):
        self.logger.info('start train')
        for epoch in range(self.num_epoch):
            self.logger.info("epoch={} started.".format(epoch + 1))     
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)           
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_batch)
            total_loss = 0.0
            bucket_size = 10
            bucket = bucket_size
            for i, sample_batched in enumerate(self.dataloader):
                if i >= self.num_batch:
                    break
                x_data = sample_batched[0].to(self.device)
                y_data = sample_batched[1].to(self.device)
                y_pred = self.model.forward(x_data)
                loss = criterion(y_pred, y_data)                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()                    
                loss.backward()
                cur_loss = loss.item()
                total_loss += cur_loss
                if 100 * (i + 1) / self.num_batch >= bucket:
                    self.logger.info("batch: {}/{}({:.2%}), samples: average loss={}; current loss={}".format(i + 1, self.num_batch, (i + 1) / self.num_batch, total_loss / (i + 1), cur_loss))
                    bucket += bucket_size
                self.logger.debug("batch: {}/{}({:.2%}), samples: average loss={}; current loss={}".format(i + 1, self.num_batch, (i + 1) / self.num_batch, total_loss / (i + 1), cur_loss))    
        self.logger.info('end train')
        return self.model
    def save_model(self, output_path):
        torch.save(self.model, output_path)
    
        