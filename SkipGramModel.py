# -*- coding: utf-8 -*-
"""
the SkipGramModel used for training Metapath2Vector
https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

Created on Tue May 12 20:19:30 2020

@author: Xiaocheng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from IIOUtils import *

class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.embeddings.weight.data, -initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.embeddings(pos_u)
        emb_v = self.embeddings(pos_v)
        emb_neg_v = self.embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)
    
    def save_embedding(self, id2node, file_path):
        embedding = self.embeddings.weight.cpu().data.numpy()
        output = []
        for id, node in enumerate(id2node):
            emb = embedding[id]            
            ret = [node, [float(x) for x in emb]]
            output.append(ret)
        write_json_lines(file_path, output)
