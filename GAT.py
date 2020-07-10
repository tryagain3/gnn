# -*- coding: utf-8 -*-
"""
Attention layer for GAT model https://arxiv.org/pdf/1710.10903.pdf

@author: Xiaocheng
"""
import torch
import torch.nn as nn

class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        """        
        
        
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output features after each attention head.
        dropout : float
            Dropout rate. Default: 0.5.
        Returns
        -------
        None.

        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = nn.Linear(input_dim, output_dim)
        self.a = nn.Linear(2 * output_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, id2features, node2neighbors):
        """        
        
        
        Parameters
        ----------
        id2features : TYPE dict Id -> Tensor (input_dim)
            the node id to node feature map
        node2neighbors : TYPE dict Id -> List of Ids
            the node and its neighbors

        Returns
        -------
        id2aggfeatures: TYPE dict Id -> Tensor (output_dim)
            the id to aggregated feature using attention
        """
        pass
        
            
class GAT(nn.Module):
    def __init__(self, num_heads, input_dim, output_dim, dropout=0.5):
        """        
        
        
        Parameters
        ----------
        num_heads : int
            the number of heads
        input_dim : int
            Dimension of input node features.
        output_dim : int
            Dimension of output features after each attention head.
        dropout : float
            Dropout rate. Default: 0.5.

        Returns
        -------
        None.

        """
        assert output_dim % num_heads == 0
        self.attentions = nn.ModuleList([GraphAttention(input_dim, output_dim // num_heads, dropout) for _ in range(num_heads)])
        
    def forward(self, id2features, node2neighbors):
        """        
        
        
        Parameters
        ----------
        id2features : TYPE dict Id -> Tensor (input_dim)
            the node id to node feature map
        node2neighbors : TYPE dict Id -> List of Ids
            the node and its neighbors

        Returns
        -------
        id2aggfeatures: TYPE dict Id -> Tensor (output_dim)
            the id to multi-head aggregated feature using attention
        """
        return torch.cat([att(id2features, node2neighbors) for att in self.attentions], dim=1)