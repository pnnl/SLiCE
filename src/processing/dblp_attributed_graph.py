
#!/usr/bin/python
import io
import sys
import json
import networkx as nx
import pandas as pd

from collections import Counter
from .attributed_graph import AttributedGraph

class DBLPGraph(AttributedGraph):
    def __init__(self, graph_file, false_edge_gen):
        """
        Base class for all graph loaders. 
        Assumes derived classes will create a networkx graph object of type
        1) nx.Graph() or nx.DiGraph()
        2) A node_attribute_dataframe per node type 
        author1 : {node_attribute_key : node_attribute_val, ......}
        author2 : {node_attribute_key : node_attribute_val, ......}

        df_publications:
        pub1 : {node_attribute_key : node_attribute_val, ......}
        pub2 : {node_attribute_key : node_attribute_val, ......}

        3) a node type dictionary {id : type}
        Note : The random walk methods implemented on top of networkx do not
        work with nx.MultiGraph() and hence MultiGraph object is not supported

        4) set of unique relations in the graph
        self.unique_relations = set()
        self.node_attr_dfs = dict()
        self.node_types = dict()
        self.G = nx.DiGraph()
        """
        super().__init__()
        self.load_graph(graph_file)
        self.create_normalized_node_id_map()
        self.get_nodeid2rowid()
        self.get_rowid2nodeid()
        self.get_rowid2vocabid_map()
        self.set_relation_id()
        self.false_edge_gen = false_edge_gen
        #json.dump(self.node_types, open("dblp_node_types.json", "w"))
        #for node_type, node_attributes in self.node_attr_dfs.items():
        #    node_attributes.to_csv(open(node_type + ".csv", "w"))
        #return
        

    def load_graph(self, graph_file, is_sample=False):
        """ 
        Implements DataLoader.load_graph() fill in
        values for
        1) self.G, 2) self.node_attrs_df, 3) self.node_types, 4)self.unique_relations
        In DBLP graph, we  consider following as nodes: {Publication, Author, Conference}
        For each node type here are the list of attributes used
        Node Attributes
        {Publication : title, field of study(fos), inverted abstract, year}
        {Author : organization}
        {Conference : field of study (infer top 5-10 from list of publications??)}


        Note : The try catch block for echa node type catches missing attributes
        for each node. Hence, the attribute references should be done before
        adding an edge for that node.
        Example : author "2796119082" has missing attribute organization. Hence,
        G.add_edge(paper_id, 2796119082, 'hasAuthor') shoudl only be called iff
        author[author_id].['org] returns successfully
        """
        venue_topic_list = dict()
        with io.open(graph_file, "r", encoding='utf8') as f:
                
            author_attr = dict()
            pub_attr = dict()
            venue_attr = dict()
            cnt = 0
            for line in f:
                if(is_sample and cnt == 1000):
                    break
                try:
                    # add publication data
                    pub_data = json.loads(line)
                    pub_id = pub_data['id']
                    pub_attr[pub_id] = {#'title' : pub_data['title'], 
                                        #'indexed_abstract' : pub_data['indexed_abstract'],
                                        ## For now we consider the top 1 fos for publication
                                        'fos' : [pub_data['fos'][ii]['name'] for ii in range(len(pub_data['fos']))][-1], 
                                        'year' : pub_data['year'],
                                        'n_citation' : pub_data['n_citation'],
                                        'type' : 'publication'
                                        }
                    self.node_types[pub_id] = 'publication'

                    # Add author data
                    pub_authors = pub_data['authors']
                    for author in pub_authors:
                        try:
                            author_id = author['id']
                            author_attr[author_id] = {'name': author['name'],
                                                   'org' : author['org'],
                                                    'type' : 'author'}
                            self.node_types[author_id] = 'author'
                            self.G.add_edge(pub_id, author_id, label='hasAuthor')
                            #self.G.add_edge(author_id, pub_id, label='inv-hasAuthor')
                        except KeyError:
                            continue

                    # Add pub-pub edges
                    try:
                        refs = pub_data['references']
                        for ref_id in refs:
                            self.G.add_edge(pub_id, ref_id, label ='cites')
                            #self.G.add_edge(ref_id, pub_id, label ='inv-cites')
                            #self.node_types[ref_id] = 'publication'
                    except KeyError:
                        pass
                        #print("Paper missing refrences", pub_id)
                    
                    # Add pub-conf edge
                    pub_venue = pub_data['venue']
                    venue_id = pub_venue['id']
                    venue_attr[venue_id] = {'title' : pub_venue['raw'], 'type':'venue'}
                    self.node_types[venue_id] = 'venue'
                    self.G.add_edge(pub_id, venue_id, label='publishedAt')
                    #self.G.add_edge(venue_id, pub_id, label='inv-publishedAt')
                    
                    venue_topic_list = self.add_topic_to_venue(venue_topic_list, pub_venue['raw'], pub_data['fos'])
                
                except KeyError:
                    print("Incomplete data for publication, ignoring : ", pub_id)
            cnt += 1

        # update venue attributes to include top 5 topics
        venue_attr = self.get_topk_venue_topics(venue_topic_list, venue_attr, k=5)
        self.node_attr_dfs = {'publication': pd.DataFrame.from_dict(pub_attr, orient='index'),
                              'author' : pd.DataFrame.from_dict(author_attr,
                                                                orient='index'),
                              'venue' : pd.DataFrame.from_dict(venue_attr,
                                                               orient='index')
                             }
        ##Dump the dataframes into csv files
        # for node_type in self.node_attr_dfs:
        #     fout = open(node_type+'.csv', 'w')
        #     self.node_attr_dfs[node_type].to_csv(fout, index=True)
        #     fout.close()

        self.unique_relations.add('hasAuthor')
        #self.unique_relations.add('inv-hasAuthor')
        self.unique_relations.add('cites')
        #self.unique_relations.add('inv-cites')
        self.unique_relations.add('publishedAt')
        #self.unique_relations.add('inv-publishedAt')
        self.cleanup_missing_references(pub_attr)
        return 

    def cleanup_missing_references(self, pub_attr):
        new_graph = nx.Graph()
        all_edges = self.G.edges()
        for u,v in self.G.edges():
            edgelabel = self.G.get_edge_data(u,v)['label']
            if ((edgelabel == 'cites' or edgelabel == "inv-cites") \
               and (u not in pub_attr or v not in pub_attr)):
                #print("Ignoring edge", u, v, edgelabel)
                continue
            else:
                new_graph.add_edge(u, v, label=edgelabel)
        self.G = new_graph
            
    def get_topk_venue_topics(self, venue_topic_list, venue_attr, k):
        """
        Given following :
            {venue title : topic list of published papers}
            {venue id : venue title}
        Generate:
            {venue id : {venue title,  top 5 topics}
        Note : venue id might be unique by year but title are common across year
        """
        for venue_id in venue_attr.keys():
            venue_title = venue_attr[venue_id]['title']
            #print(venue_title, venue_topic_list[venue_title])
            venue_topics = Counter(venue_topic_list[venue_title])
            sorted_topics = sorted(venue_topics.items(), key=lambda kv:kv[1])
            #print(venue_title, sorted_topics[-k:])
#             venue_attr[venue_id]['topics'] = sorted_topics[-k:]
#             print(venue_title, sorted_topics[-k:])
            venue_attr[venue_id]['topics'] = sorted_topics[-k:][-1][0]
        return venue_attr


    def get_wide_cols(self):
        return {'author' : ['type','org'],
                'publication': ['type', 'fos'],
                'venue': ['type', 'topics']
               }

    def get_continuous_cols(self):
        return {'author': None,
                'publication': ['year', 'n_citation'],
                'venue': None
               }
    
    def add_topic_to_venue(self, venue_topic_dictionary, venue, topic_list):
        if venue not in venue_topic_dictionary:
            venue_topic_dictionary[venue] = list()

        for x in topic_list:
            venue_topic_dictionary[venue].append(x['name'])
        return venue_topic_dictionary
