# -*- coding: utf-8 -*-
"""
class represent a Hetegeneous Graph (Undirected Graph)

Created on Sat May  9 12:05:10 2020

@author: Xiaocheng
"""


import networkx as nx

TYPE_ATTR = 'type'
WEIGHT_ATTR = 'weight'

class HeterogeneousGraph:
    def __init__(self):
        self.g = nx.MultiGraph()
        
    def add_edge(self, u, v, t, w = 1):
        """        
        add type edge between node u and v, with edge_type t, with edge_weight w
        """
        if u not in self.g:
            raise Exception('add_edge failed:  node "{}" not in graph'.format(u))
        
        if v not in self.g:
            raise Exception('add_edge failed:  node "{}" not in graph'.format(v)) 
        
        i = self.g.add_edge(u, v)
        edge_attributes = self.g[u][v][i]
        edge_attributes[TYPE_ATTR] = t
        edge_attributes[WEIGHT_ATTR] = 1
        
    def add_node(self, u, t):
        """
        add node u with type t
        """
        self.g.add_node(u)
        self.g.nodes[u][TYPE_ATTR] = t
    
    def add_edge_feature(self, u, v, t, features):
        """
        add features for edge (u, v, t)
        """
        if u not in self.g:
            raise Exception('add_edge_feature failed:  node "{}" not in graph'.format(u))
        
        if v not in self.g:
            raise Exception('add_edge_feature failed:  node "{}" not in graph'.format(v)) 
        
        if v not in self.g[u]:
            raise Exception('add_edge_feature failed:  node "{}" has no edge connected to "{}" in graph'.format(u, v)) 
        
        edges = [i for i in self.g[u][v] if self.g[u][v][i][TYPE_ATTR] == t]
        if not edges:
            raise Exception('add_edge_feature failed:  node "{}" has no edge connected to "{}" in graph with edge type "{}"'.format(u, v, t)) 
        
        i = edges[0]
        attr = self.g[u][v][i]
        
        for f, v in features.items():
            if f == TYPE_ATTR:
                raise Exception('add_edge_feature: feature name "{}" is same with edge type name for edge ("{}","{}","{}"), please change the feature name'.format(f, u, v, t))
            if f == WEIGHT_ATTR:
                raise Exception('add_edge_feature: feature name "{}" is same with edge weight name for edge ("{}","{}","{}"), please change the feature name'.format(f, u, v, t))
            attr[f] = v
   
    
    def add_node_feature(self, u, features):
        """
        add features for node u
        """
        if u not in self.g:
            raise Exception('add_node_feature: node "{}" not in graph'.format(node_id))
        
        attr = self.g.nodes[u]
        for f, v in features.items():
            if f == TYPE_ATTR:
                raise Exception('add_node_feature: feature name "{}" is same with node type name for node "{}", please change the feature name'.format(f, u))
            attr[f] = v
        
    
    def get_node_type(self, node_id):
        """
        get the type of the given node_id
        """
        if node_id not in self.g:
            raise Exception('get_node_type:  node "{}" not in graph'.format(node_id))
        return self.g.nodes[node_id][TYPE_ATTR]
    
    def get_neighbours(self, u, node_type = None, edge_type = None):
        """
        get all neighbours that its type is node_type and the edge connected with u is edge_type
        """
        if u not in self.g:
            raise Exception('get_neighbours:  node "{}" not in graph'.format(node_id))
        
        neis = self.g[u]
        result = []
        for v in neis:
            if node_type != None and self.g.nodes[v][TYPE_ATTR] != node_type:
                continue
            if edge_type != None and all(attr[TYPE_ATTR] != edge_type for attr in self.g[u][v]):
                continue
            result.append(v)
        return result
    
    def node_number(self, node_type = None):
        """
        get node number for given node_type, if node_type == None, return all node number
        """
        if node_type == None:
            return self.g.number_of_nodes()
        else:
            return len([u for u in self.g.nodes if self.g.nodes[u][TYPE_ATTR] == node_type])
        
    def edge_number(self, edge_type = None):
        """
        get edge number for given edge_type, if edge_type == None, return all edge number
        """
        if edge_type == None:
            return self.g.number_of_edges()
        else:
            cnt = 0
            for u, v in self.g.edges():
                edges = self.g[u][v]
                for i in edges:
                    if edges[i][TYPE_ATTR] == edge_type:
                        cnt += 1
                        break
            return cnt
    
    def get_nodes(self):
        """
        return all nodes in this graph
        """
        for node in self.g.nodes:
            yield node
        