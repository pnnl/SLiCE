
#!/usr/bin/python
import io
import os
import json
import networkx as nx
import pandas as pd
from collections import Counter
from src.processing.attributed_graph import AttributedGraph

class FreebaseGraph(AttributedGraph):
    def __init__(self, main_dir, false_edge_gen):
        """
        Assumes derived classes will create a networkx graph object along with
        following 
        1) self.unique_relations = set()
        2) self.node_attr_dfs = dict()
        3) self.node_types = dict()
        4) self.G  = nx.Graph()
        """
        super().__init__()
        self.main_dir = main_dir
        self.false_edge_gen = false_edge_gen
        self.load_graph(main_dir + "/train.txt")
        self.load_graph(main_dir + "/valid.txt")
        self.load_graph(main_dir + "/test.txt")
        self.create_normalized_node_id_map()

        attributes_file = ''
        #get product attributes
        self.node_attr = dict()
        self.node_attr_list = []
        if(attributes_file == ""):
            for node_id in self.G.nodes():
                self.node_attr[node_id] = dict()
                self.node_attr[node_id]['defattr'] = 'True'
            self.node_attr_list = ['defattr']
        else:
            raise NotImplementedError

        self.node_attr_dfs = {
            'deftype': pd.DataFrame.from_dict(self.node_attr, orient='index')
        }

        self.get_nodeid2rowid()
        self.get_rowid2nodeid()
        self.get_rowid2vocabid_map()
        self.set_relation_id()

    
    def load_graph(self, filename):
        """ 
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.node_attr_dfs
        3) self.unique_relations
        4) self.node_types
        """
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                if(len(arr) >= 3):
                    source = arr[0]
                    edge_type = arr[1]
                    target = arr[2]
                    self.G.add_edge(source, target, label=edge_type)
                    self.unique_relations.add(edge_type)
                    self.node_types[source] = 'deftype'
                    self.node_types[target] = 'deftype'
                else:
                    print("Ignoring line : ", line)
        return

    def load_test_edges(self, filename):
        true_and_false_edges = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split("\t")
                if(len(arr) >= 3):
                    source = arr[0]
                    relation = arr[1]
                    target = arr[2]
                    if(len(arr) == 4):
                        true_and_false_edges.append((relation, source, target,
                                                     arr[3]))
                    else:
                        true_and_false_edges.append((relation, source, target,
                                                     "1"))
                else:
                    print("Ignoring line : ", line)
        return true_and_false_edges

    def generate_link_prediction_dataset(self, outdir, fr_valid_edges,
                                          fr_test_edges):
        print("In freebase link prediction generation")
        
        self.train_edges = self.load_test_edges(self.main_dir + "/train.txt")
        self.valid_edges = self.load_test_edges(self.main_dir + "/valid.txt")
        self.test_edges = self.load_test_edges(self.main_dir + "/test.txt")
        false_edge_gen = self.false_edge_gen

        if(false_edge_gen == 'pattern'):
            self.train_edges = self.generate_false_edges3(self.train_edges)
            self.valid_edges = self.generate_false_edges3(self.valid_edges)
            self.test_edges = self.generate_false_edges3(self.test_edges)
        elif(false_edge_gen == 'double'):
            self.train_edges = self.generate_false_edges2(self.train_edges)
            self.valid_edges = self.generate_false_edges2(self.valid_edges)
            self.test_edges = self.generate_false_edges2(self.test_edges)
        else:
            #self.train_edges = self.generate_false_edges(self.train_edges)
            self.valid_edges = self.generate_false_edges(self.valid_edges)
            self.test_edges = self.generate_false_edges(self.test_edges)
        return self.train_edges, self.valid_edges, self.test_edges

    """
    def load_node_attributes(self, filename):
        node_attr_list = []
        with open(filename, "r") as f:
            line = f.readline()
            for line in f:
                arr = line.strip().split()
                graph_node_id = arr[0]
                node_attr = [float(x) for x in arr[1:]]
                for i in range(len(node_attr)):
                    self.node_attr[graph_node_id] = dict()
                    node_attr_key = "attr_"+str(i)
                    self.node_attr[graph_node_id][node_attr_key] = node_attr[i]
                    node_attr_list.append(node_attr_key)
        return self.node_attr, list(set(node_attr_list))
    """

    def get_continuous_cols(self):
        return {'deftype' : None}

    def get_wide_cols(self):
        return {'deftype' : self.node_attr_list}
