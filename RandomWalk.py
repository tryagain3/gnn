# -*- coding: utf-8 -*-
"""
The class used for generate random walk (support metapath random walk and none metapath random walk)
when the meta schema is set, there is assumption that the there is no multi edge relation.
basically, there is only one edge type bewteen the same node type.

e.g. there are two types of nodes: Author and Paper, we only have one edge type between these two node type,
which means the person is the author of the paper

support two modes:
    if meta_path_schema is set, will generate random walk based on meta_path_schema
    if meta_path_schema is not set, will generate random walk ingore the edge type and node type
    
Created on Wed Apr 29 20:25:45 2020

@author: Xiaocheng
"""
import random
import networkx as nx
import numpy as np
from Logger import DefaultLogger
from HeterogeneousGraph import HeterogeneousGraph

class RandomWalk:
    def __init__(self, g, logger=DefaultLogger(True)):
        self.g = g    
        self.logger = logger

    def walk(self, walk_length, start, schema=None):
        """
        walk_length: the length of the walk
        start: the start node
        meta_path_schema: the path schema, should be V1 -> V2 -> V3 .. -> Vl
        meta-paths are commonly used in a symmetric way, that is, its rst node type V1 is the same with the last one Vl
        plesae refer https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf for more understanding of meta_path
        """
        rand = random.Random()
        walk = [start]
        schema_iter = schema.get_iter()
        next(schema_iter)
        while len(walk) < walk_length:
            cur = walk[-1]
            node_type = edge_type = None
            if schema == None:
                node_type, edge_type = next(schema_iter)
            candidates = self.g.get_neighbours(cur, node_type, edge_type)               
            if candidates:
                walk.append(rand.choice(candidates))
            else:
                break
        return walk
    
    def random_walk(self, num_walk, walk_length, schemas=None):
        """
        num_walk: the number of walks needed to be generated for each schema/node
        walk_length: the length of the walk
        start: the start node
        meta_path_schemas: the list of meta_path_schema
        plesae refer https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf for more understanding of meta_path    
        """
        self.logger.info('start generate random walks')
        walks = []
        nodes = list(self.g.get_nodes())
        if schemas:
            self.logger.info('schemas count={}'.format(len(schemas)))
            self.logger.info('list all schemas')
            for schema in schemas:
                self.logger.info(str(schema))
                
        for walk_iter in range(num_walk):
            random.shuffle(nodes)
            for node in nodes:
                if schemas is None:
                    walks.append(self.walk(walk_length=walk_length, start=node))
                else:
                    for schema in schemas:
                        if schema.get_start_type() == self.g.get_node_type(node):
                            walks.append(self.walk(walk_length=walk_length, start=node, schema=schema))
        self.logger.info('generate {} randow walks'.format(len(walks)))
        self.logger.info('end generate random walks')
        return walks
    