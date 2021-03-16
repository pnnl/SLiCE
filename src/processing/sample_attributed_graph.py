import networkx as nx
import pandas as pd
import torch
from .attributed_graph import AttributedGraph

class SampleGraph(AttributedGraph):
    def __init__(self, edge_file, node_attr_file):
        super().__init__()
        self.load_graph(edge_file, node_attr_file)

    def load_graph(self, edge_file, node_attr_file):
        with open(edge_file) as f:
            lines = f.readlines()
            for line in lines[1:]:
                arr = line.strip().split(',')
                self.G.add_edge(arr[0], arr[1], label=arr[2])
                self.G.add_edge(arr[1], arr[0], label="inv-" + arr[2])
                self.unique_relations.add(arr[2])
                self.unique_relations.add("inv-" + arr[2])
       
       #TODO update for dataframe
        node_attrs = pd.read_csv(node_attr_file)
        for i in range(len(node_attrs)):
            self.node_attrs[node_attrs.at[i, 'node_id']] = node_attrs.iloc[i, 1:4]
            self.nodeid[node_attrs.at[i, 'node_id']] = i

