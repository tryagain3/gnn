# -*- coding: utf-8 -*-
"""
logistic regression for muli classification
https://en.wikipedia.org/wiki/Logistic_regression

Created on Tue May 19 20:53:09 2020

@author: Xiaocheng
"""


import torch
import torch.nn as nn
from torch.nn import init

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_size, output_size):
        """
        input_size: input feature dimension
        output_size: the output data size which is the number of classes

        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        the input x should have the same dimension with self.linear
        """
        return torch.sigmoid(self.linear(x))