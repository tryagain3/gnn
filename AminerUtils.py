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
from torch.utils.data import DataLoader
import os
import json
from LogisticRegressionModelTrainer import LogisticRegressionModelTrainer
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch

AUTHOR_NODE_TYPE = 'A'
CONF_NODE_TYPE = 'C'

def build_node_id(raw_id, node_type):
    return '{}_{}'.format(node_type, raw_id)

def decode_raw_id(id):
    items = id.split('_')
    return itesm[1] if len(items) > 1 else items[0]

def parse_file(file_path, g, node_type):
    result = defaultdict(list)
    for line in get_data_row(file_path):
        line = line if line and line[-1]!='\n' else line[:-1]
        if not line: continue
        x, y = line.split('\t')
        result[x].append(y)
        g.add_node(build_node_id(y, node_type), node_type)
    return result

def parse_schemas_file(file_path):
    ret = []
    for line in get_data_row(file_path):
        line = line if line and line[-1]!='\n' else line[:-1]
        if not line: continue
        ret.append(MetaPathSchema(line))
    return ret


class ListDataSet(IterableDataset):    
    def __init__(self, data):
        """
        initialize torch dataset using data_provider which is a SkipGramDataProvider
        """
        self.data = data
        self.i = 0
               
    def __iter__(self):
        while True:
            yield self.data[self.i]
            self.i = (self.i + 1) % len(self.data)
    
    @staticmethod
    def collate(batches):
        x_data = [x for x, _ in batches]
        y_data = [v for _, y in batches]
        return torch.FloatTensor(x_data), torch.LongTensor(y_data)

        

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
                g.add_edge(build_node_id(a_id, AUTHOR_NODE_TYPE), build_node_id(c_id, CONF_NODE_TYPE) , '')    
    
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
    #gen_training_data(input_folder, walks, negative_sample_size, window_size, logger)

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
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=dataset.collate)
    samples = []
    i = 0
    for data in dataloader:
        if i >= 100:
            break
        samples.extend(data)
        i += 1
    logger.info('total generate {} training samples'.format(len(samples)))
    logger.debug('show top 5 generated samples: {}'.format(samples[:5]))    
    file_path = os.path.join(input_folder, 'train_samples.txt')
    logger.info('write samples to file: {}'.format(file_path))
    write_json_lines(file_path, [get_vale_from_tensor(item) for item in samples])   
    file_path = os.path.join(input_folder, 'node2id.txt')
    write_lines(file_path, ['{}:{}'.format(node, id) for node, id in dataset.data_provider.node2id.items()])
    logger.info('end load data')
def get_vale_from_tensor(item):
    x, y, z = item[0], item[1], item[2]
    return [str(x.numpy()), str(y.numpy()), str(z.numpy())]

def load_conf_data_for_classification(input_folder=r'D:\data\net_aminer', logger=DefaultLogger(True)):
    """
    should contain the following data
    02/22/2017  04:18 PM         4,253,730 googlescholar.8area.author.label.txt
    02/22/2017  04:18 PM             2,366 googlescholar.8area.venue.label.txt
    08/29/2017  09:37 PM        38,616,219 id_author.txt
    08/29/2017  09:37 PM            73,552 id_conf.txt
    05/16/2020  06:38 PM     2,848,353,597 node_embedding.txt
    """
    
    logger.info('load conf labels')
    conf_labels = [line[:-1].split(' ') for line in get_data_row(os.path.join(input_folder, 'googlescholar.8area.venue.label.txt'))]
    logger.info('total load {} conf labels'.format(len(conf_labels)))    
    
    logger.info('load id -> conf maps')
    raw = [line[:-1].split('\t') for line in get_data_row(os.path.join(input_folder, 'id_conf.txt'))]
    conf2id = {y:x for x, y in raw}
    logger.info('total load {} conf -> id'.format(len(conf2id)))
    
    num_category = 8    
    logger.info('build conf id to labels ')
    cnt = 0
    dup_id_cnt = 0
    conf2labels = {}
    for conf_name, label in conf_labels:
        if conf_name in conf2id:
            id = conf2id[conf_name]
            if id not in conf2labels:
                conf2labels[id] = int(label) - 1
            else:
                dup_id_cnt += 1
        else:
            cnt += 1    
    logger.info('#failed to find conf name in conf2id map: {}'.format(cnt))
    logger.info('#dup id with duplicate labels: {}'.format(dup_id_cnt))
    return conf2labels

def load_author_data_for_classification(input_folder=r'D:\data\net_aminer', logger=DefaultLogger(True)):
    """
    should contain the following data
    02/22/2017  04:18 PM         4,253,730 googlescholar.8area.author.label.txt
    02/22/2017  04:18 PM             2,366 googlescholar.8area.venue.label.txt
    08/29/2017  09:37 PM        38,616,219 id_author.txt
    08/29/2017  09:37 PM            73,552 id_conf.txt
    05/16/2020  06:38 PM     2,848,353,597 node_embedding.txt
    """
    
    logger.info('load author labels')
    author_labels = [line[:-1].split(' ') for line in get_data_row(os.path.join(input_folder, 'googlescholar.8area.author.label.txt'))]
    logger.info('total load {} author labels'.format(len(author_labels)))    
    
    logger.info('load id -> author maps')
    raw = [line[:-1].split('\t') for line in get_data_row(os.path.join(input_folder, 'id_author.txt'))]
    author2id = {y:x for x, y in raw}
    logger.info('total load {} conf -> id'.format(len(author2id)))
    
    num_category = 8    
    logger.info('build author id to labels ')
    cnt = 0
    dup_id_cnt = 0
    author2labels = {}
    for author_name, label in author_labels:
        if author_name in author2id:
            id = author2id[author_name]
            if id not in author2labels:
                author2labels[id] = int(label) - 1
            else:
                dup_id_cnt += 1
        else:
            cnt += 1    
    logger.info('#failed to find author name in author2id map: {}'.format(cnt))
    logger.info('#dup id with duplicate labels: {}'.format(dup_id_cnt))
    return author2labels
def load_embedding(input_folder=r'D:\data\net_aminer', logger=DefaultLogger(True)):
    """
    should contain the following data
    02/22/2017  04:18 PM         4,253,730 googlescholar.8area.author.label.txt
    02/22/2017  04:18 PM             2,366 googlescholar.8area.venue.label.txt
    08/29/2017  09:37 PM        38,616,219 id_author.txt
    08/29/2017  09:37 PM            73,552 id_conf.txt
    05/16/2020  06:38 PM     2,848,353,597 node_embedding.txt
    """
    pass
    
def train_evaluate_conf_classification(conf2labels, embedding, rate=0.2, batch_size=32, num_epoch=3, num_batch=100000, initial_lr=0.001, logger=DefaultLogger(True)):   
    num_category = 8
    emb_dimension = 128
    logger.info('map embedding feature to labels')
    data = []
    cnt = 0
    for raw_id, label in conf2labels.items():
        embedding_id = build_node_id(raw_id, CONF_NODE_TYPE)
        if embedding_id in embedding:
            data.append([embedding[embedding_id], label])
        else:
            cnt += 1
    logger.info('#confs with no embedding map: {}'.format(cnt))    
    train, test = split(data, rate=rate)
    
    data_set = ListDataSet(train)
    data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=self.dataset.collate)
    trainer = LogisticRegressionModelTrainer(dataloader=dataloader, 
                                             input_size=emb_dimension, 
                                             output_size=num_category, 
                                             num_epoch=num_epoch, 
                                             num_batch=num_batch, 
                                             initial_lr=initial_lr,
                                             logger=logger)
    model = trainer.train()
    tp, fp, fn = [0] * numc_category, [0] * numc_category, [0] * numc_category
    with torch.no_grad():
        model.eval()
        x_data = [x for x, _ in test]
        y_data = [y for x, _ in test]
        outputs = model(x_data)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(numc_category):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    pass
        
        
        

    
    
    
    
if __name__ == '__main__':
    #load_data(input_folder=r'D:\data\net_aminer_test') 
    #load_data2(input_folder=r'D:\data\net_aminer_test', logger=DefaultLogger(True)) 
    #load_data(input_folder=r'D:\data\net_aminer')    
    #load_data2(input_folder=r'D:\data\net_aminer')     
    
    """
    input_folder = r'D:\data\net_aminer'
    logger = DefaultLogger(True)
    conf2labels = load_conf_data_for_classification(input_folder=input_folder, logger=logger)     
    author2labels = load_author_data_for_classification(input_folder=input_folder,logger=logger)     
    dup_ids = set([build_node_id(x,CONF_NODE_TYPE) for x in conf2labels.keys()]) & set([build_node_id(x, AUTHOR_NODE_TYPE) for x in author2labels.keys()])
    logger.info('#duplicate id between author and conf: {}'.format(len(dup_ids)))
    logger.info('duplicate ids: {}'.format(dup_ids))
    """
        
        
        
