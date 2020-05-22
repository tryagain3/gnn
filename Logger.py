# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:45:08 2020

@author: Xiaocheng
"""

from datetime import datetime

class DefaultLogger:
    def __init__(self, debug = True):
        self.debug_flag = debug
    def info(self, message):   
        self.__print__('info', message)        
    def error(self, message):
        self.__print__('error', message)        
    def debug(self, message):
        if self.debug_flag:
            self.__print__('debug', message)         
    def __print__(self, tag, message):
        print('[{}][{}]{}'.format(datetime.now(), tag, message))
        
class FileLogger:
    def __init__(self, file_path, debug = True):
        self.debug_flag = debug
        self.file_path = file_path
    def info(self, message):   
        self.__print__('info', message)        
    def error(self, message):
        self.__print__('error', message)        
    def debug(self, message):
        if self.debug_flag:
            self.__print__('debug', message)         
    def __print__(self, tag, message):
        line = '[{}][{}]{}\n'.format(datetime.now(), tag, message)
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(line)