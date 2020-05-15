# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:57:44 2020

Utils class to process Aminer data

AMiner Computer Science (CS) Data: The CS dataset consists of 1,693,531 computer scientists and 3,194,405 papers from 3,883 computer science venues---both conferences and journals---held until 2016. We construct a heterogeneous collaboration network, in which there are three types of nodes: authors, papers, and venues.

Citation: Jie Tang, Jing Zhang, Limin Yao, Juanzi Li, Li Zhang, and Zhong Su. 2008. ArnetMiner: Extraction and Mining of Academic Social Networks. In KDD'08. 990–998.

download link: https://www.dropbox.com/s/1bnz8r7mofx0osf/net_aminer.zip?dl=0&file_subpath=%2Fnet_aminer

reference https://ericdongyx.github.io/metapath2vec/m2v.html

02/22/2017  04:18 PM         4,253,730 googlescholar.8area.author.label.txt
02/22/2017  04:18 PM             2,366 googlescholar.8area.venue.label.txt
08/29/2017  09:37 PM        38,616,219 id_author.txt
08/29/2017  09:37 PM            73,552 id_conf.txt
05/08/2020  03:02 PM                 3 meta_path_schemas.txt
01/04/2017  11:10 PM       262,635,328 paper.txt
01/04/2017  11:10 PM       135,169,967 paper_author.txt
01/04/2017  11:10 PM        39,433,500 paper_conf.txt

@author: Xiaocheng
"""
from collections import defaultdict
import networkx as nx
from IIOUtils import *
from Logger import DefaultLogger
from HeterogeneousGraph import HeterogeneousGraph
from MetaPathSchema import MetaPathSchema
from SkipGramDataSet import SkipGramDataSet
from SkipGramDataProvider import SkipGramDataProvider
from RandomWalk import RandomWalk
import os
import json

AUTHOR_NODE_TYPE = 'A'
CONF_NODE_TYPE = 'C'

def parse_file(file_path, g, node_type):
    result = defaultdict(list)
    for line in get_data_row(file_path):
        line = line if line and line[-1]!='\n' else line[:-1]
        if not line: continue
        x, y = line.split('\t')
        result[x].append(y)
        g.add_node(y, node_type)
    return result

def parse_schemas_file(file_path):
    ret = []
    for line in get_data_row(file_path):
        line = line if line and line[-1]!='\n' else line[:-1]
        if not line: continue
        ret.append(MetaPathSchema(line))
    return ret
        

def load_data(input_folder=r'D:\data\net_aminer', num_walk=10, walk_length=5, negative_sample_size=5, window_size=5, logger=DefaultLogger(True)):
    """
    the folder should contain the following files
        05/08/2020  03:02 PM                 3 meta_path_schemas.txt
        01/04/2017  11:10 PM       135,169,967 paper_author.txt
        01/04/2017  11:10 PM        39,433,500 paper_conf.txt
    """
    logger.info('start load data')
    g = HeterogeneousGraph()
    
    logger.info('load data: paper_author.txt')
    p_a = parse_file(os.path.join(input_folder, 'paper_author.txt'), g, AUTHOR_NODE_TYPE)
    
    logger.info('load data: paper_conf.txt')
    p_c = parse_file(os.path.join(input_folder, 'paper_conf.txt'), g, CONF_NODE_TYPE)
    
    author_cnt = g.node_number(AUTHOR_NODE_TYPE)
    conf_cnt = g.node_number(CONF_NODE_TYPE)
    logger.info('read {} author nodes'.format(author_cnt))
    logger.info('read {} conf nodes'.format(conf_cnt))
    
    logger.info('read {} papers from paper_author.txt'.format(len(p_a)))
    logger.info('read {} papers from paper_conf.txt'.format(len(p_c)))
    
    logger.info('average #paper per author: {}'.format(len(p_a) / author_cnt))
    logger.info('average #paper per conf: {}'.format(len(p_c) / conf_cnt))
    
    for p_id, a_ids in p_a.items():
        for a_id in a_ids:
            for c_id in p_c[p_id]:
                # no edge type for AMiner graph
                g.add_edge(a_id, c_id, '')    
    
    logger.info('get {} Author-Conf edges'.format(g.edge_number()))
    walk_generator = RandomWalk(g=g, logger=logger)
    
    logger.info('load meta path schemas')
    schemas = parse_schemas_file(os.path.join(input_folder, 'meta_path_schemas.txt'))
    logger.info('read {} metapath scehams'.format(len(schemas)))
    logger.debug('log each schema')
    for i, s in enumerate(schemas):
        logger.debug('{}:"{}"'.format(i, s))
    
    logger.info('generate random walks with configration: num_walk={}, walk_length={}'.format(num_walk, walk_length))
    walks = walk_generator.random_walk(num_walk=num_walk, walk_length=walk_length, schemas=schemas)
    logger.info('total generate {} random walks  '.format(len(walks)))
    
    file_path = os.path.join(input_folder, 'random_walks.txt')
    logger.info('write walks to file: {}'.format(file_path))
    write_json_lines(file_path, walks)
    gen_training_data(input_folder, walks, negative_sample_size, window_size, logger)

def load_data2(input_folder=r'D:\data\net_aminer', negative_sample_size=5, window_size=5, logger=DefaultLogger(True)):
    walk_file_path = os.path.join(input_folder, 'random_walks.txt')
    logger.info('load walk file {}'.format(walk_file_path))
    walks = [json.loads(line) for line in get_data_row(walk_file_path)]
    logger.info('total load {} random walks  '.format(len(walks)))
    gen_training_data(input_folder, walks, negative_sample_size, window_size, logger)

def gen_training_data(input_folder, walks, negative_sample_size, window_size, logger):        
    logger.info('generate training samples')
    data_provider = SkipGramDataProvider(walks=walks, negative_sample_size=negative_sample_size, window_size=window_size, logger=logger)
    dataset = SkipGramDataSet(data_provider)
    samples = [dataset.__iter__() for _ in range(10000)]
    logger.info('total generate {} training samples'.format(len(samples)))
    logger.debug('show top 5 generated samples: {}'.format(samples[:5]))    
    file_path = os.path.join(input_folder, 'train_samples.txt')
    logger.info('write samples to file: {}'.format(file_path))
    write_json_lines(file_path, samples)   
    file_path = os.path.join(input_folder, 'node2id.txt')
    write_lines(file_path, ['{}:{}'.format(node, id) for node, id in dataset.node2id.items()])
    logger.info('end load data')
    
    
if __name__ == '__main__':
    #load_data(input_folder=r'D:\data\net_aminer_test') 
    #load_data2(input_folder=r'D:\data\net_aminer_test', logger=DefaultLogger(True)) 
    #load_data(input_folder=r'D:\data\net_aminer')    
    load_data2(input_folder=r'D:\data\net_aminer')     
        
        
        
