# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:45:08 2020

@author: Xiaocheng
"""

from datetime import datetime

class DefaultLogger:
    def __init__(self, debug = True):
        self.debug = debug
    def info(self, message):   
        self.__print__('info', message)        
    def error(self, message):
        self.__print__('error', message)        
    def debug(self, message):
        if self.debug:
            self.__print__('debug', message)         
    def __print__(self, tag, message):
        print('[{}][{}]{}'.format(datetime.now(), tag, message))