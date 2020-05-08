# -*- coding: utf-8 -*-
"""
The class represent the meta_path schema

V1->V2->V3... -> Vl

meta-paths are commonly used in a symmetric way, that is, its rst
node type V1 is the same with the last one Vl

the length of meta-paths should be greater or equal to 3

Created on Mon May  4 21:04:23 2020

@author: Xiaocheng
"""


class MetaPathSchema:
    def __init__(self, schema_str):
        self.schema = schema_str.split('->')
        assert len(self.schema) >= 3
        assert self.schema[0] == self.schema[-1]
    def get_start_type(self):
        return self.schema[0]
    def __repr__(self):
        return '->'.join(self.schema)
    def get_iter(self):
        return MetaPathSchemaIter(self)

class MetaPathSchemaIter:
    def __init__(self, meta_path_schema):
        self.meta_path_schema = meta_path_schema
        self.i = 0
    def __next__(self):
        ret = self.meta_path_schema.schema[self.i]
        self.i = (self.i + 1) % len(self.meta_path_schema)
        return ret
        
    