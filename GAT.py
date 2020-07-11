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
        
    def forward(self, features, nodes, neighbors):
        """        
        
        
        Parameters
        ----------
        features : Tensor with m * input_dim
            reprsent the features for all the nodes (including nodes and its neighbors)
        nodes : 
            list, indicate the target nodes postions in the features
        neighbors:
            list of int list, indicate the corresponding negghbors for the input nodes in features
        Returns
        -------
        id2aggfeatures: Tensor (len(nodes) * output_dim)
            the output features for the nodes
        """
        # step1 apply w for each input feature
        h = self.w(features)
        
        # step2 cat nbr_h and self_h
        nbr_h = [h[nbrs] for nbrs in neighbors]
        self_h = [h[x] for x in [[n] * len(nbrs) for n, nbrs in zip(nodes, neighbors)]]
        cat_h = torch.cat((self_h, nbr_h), dim=1)
        
        # step3 leakyrelu + self.a(cat_h)
        e = self.leakyrelu(self.a(cat_h))
        
        # step4 softmax for all neighbors
        # position of the nodes and its neighbors 
        pos = [0]        
        for nbr in neighbors:
            pos.append(pos[-1] + len(nbr))
        alpha = [self.softmax(e[pos[i - 1] : pos[i]]) for i in range(1, len(pos))]
        alpha = torch.cat(alpha, dim=0)
        alpha = alpha.squeeze(1)
        
        # step5 dropout 
        alpha = self.dropout(alpha)

        # step6 alpha * h_neibors        
        indices = torch.LongTensor([[u, v] for (u, nbrs) in zip(nodes, neighbors) for v in nbrs]).t()
        n = features.shape[0]        
        # https://pytorch.org/docs/stable/sparse.html
        adj = torch.sparse.FloatTensor(indices, alpha, torch.Size([n, n]))
        output = torch.sparse.mm(adj, h)[nodes]
        return output
        
            
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