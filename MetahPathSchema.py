# -*- coding: utf-8 -*-
"""
The class represent the meta_path schema

V1->E1->V2->E2->V3... ->EL-1->Vl

meta-paths are commonly used in a symmetric way, that is, its rst
node type V1 is the same with the last one Vl

the length of meta-paths should be greater or equal to 3

Created on Mon May  4 21:04:23 2020

@author: Xiaocheng
"""


class MetaPathSchema:
    def __init__(self, schema_str):
        self.schema = schema_str.split('->')
        # schema should have at lest 3 node types and 2 edge types
        # the total lengh of the schema should be an odd number
        assert len(self.schema) >= 5 and (len(self.schema) & 1)
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
        node_type = self.meta_path_schema.schema[self.i]
        if self.i == 0:
            edge_type = self.meta_path_schema.schema[len(self.meta_path_schema.schema) - 2]
        else:
            edge_type = self.meta_path_schema.schema[self.i - 1]
        self.i = (self.i + 2) % len(self.meta_path_schema)
        return node_type, edge_type
        
    