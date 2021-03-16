
#!/usr/bin/python
import io
import json
import networkx as nx
import pandas as pd
from collections import Counter
from .attributed_graph import AttributedGraph

class MetaQAGraph(AttributedGraph):
    def __init__(self, init_dir, question_dir=None):
        """
        Assumes derived classes will create a networkx graph object along with
        following 
        1) self.unique_relations = set()
        2) self.node_attr_dfs = dict()
        3) self.node_types = dict()
        4) self.G  = nx.Graph()
        """
        super().__init__()
        self.load_graph(init_dir + "/kb.txt")


    def load_graph(self, edgefile, node_attr_file=None):
        """ 
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.node_attr_dfs
        3) self.unique_relations
        4) self.node_types
        
        MetaQA: node types: Movie, Director, Writer

        MetaQA Attribute details per node type:
        Movie attributes : has_genre, imdb_rating, has_tags, language, year
        Director/Writer attributes : has_genre
        """
        self.movie_attr = dict()
        self.director_attr = dict()
        self.writer_attr = dict()

        self.director_to_movies = dict()
        self.writer_to_movies = dict()


        with io.open(edgefile, "r", encoding='utf8') as f:
            for line in f:
                arr = line.strip().split('|')
                #self.add_node_types(arr[0], arr[1], arr[2])
                if (arr[1] == 'directed_by' or arr[1] == 'directed_by'):
                    self.G.add_edge(arr[0], arr[2], label=arr[1])
                    #self.G.add_edge(arr[2], arr[0], label="inv-"+arr[1])
                    self.unique_relations.add(arr[1])
                    #self.unique_relations.add("inv-" + arr[1])
                self.update_attr(arr[0], arr[1], arr[2])
        self.director_attr = self.get_topk_attributes(self.director_to_movies, self.movie_attr, 'director')
        self.writer_attr = self.get_topk_attributes(self.writer_to_movies, self.movie_attr, 'writer')
        
        self.node_attr_dfs = {'movie': pd.DataFrame.from_dict(self.movie_attr, orient='index'),
                              'director': pd.DataFrame.from_dict(self.director_attr, orient='index'),
                              'writer': pd.DataFrame.from_dict(self.writer_attr, orient='index')
                             }
        self.set_relation_id()


    #def add_node_types(self, source, edge_label, target):
    def update_attr(self, source, edge_label, target):
        """
        Infers node type of source and target from edge type for MetaQA
        Note : this function was used once, to create and save the node types dictionary,
        this is not used now, kept for reference only
        """
        self.node_types[source] = "movie"
        if(edge_label == "directed_by"):
            self.node_types[target] = "director"
            if target in self.director_to_movies:
                self.director_to_movies[target].append(source)
            else:
                self.director_to_movies[target] = [source]
        elif(edge_label == "written_by"):
            self.node_types[target] = "writer"
            if target in self.writer_to_movies:
                self.writer_to_movies[target].append(source)
            else:
                self.writer_to_movies[target] = [source]
        else:
            if (source not in self.movie_attr):
                self.movie_attr[source] = dict()
                self.movie_attr[source]['title'] = source
            self.movie_attr[source][edge_label] = target
            self.movie_attr[source]['type'] = 'movie'

    def get_topk_attributes(self, person_movie_dict, movie_attrs, node_type, k=5):

        result_attr = dict()
        for person, movie_list in person_movie_dict.items():
            all_genres_person = Counter([movie_attrs[movie]['has_genre'] for movie in movie_list if (movie in movie_attrs and 'has_genre' in movie_attrs[movie])])
            topk_genres_person = sorted(all_genres_person.items(), key=lambda kv:kv[1])
            if(len(topk_genres_person) > 0):
                result_attr[person] = dict()
                result_attr[person]['name'] = person
                result_attr[person]['has_genre'] = topk_genres_person[-1][0]
                result_attr[person]['type'] = node_type
        return result_attr

    def get_continuous_cols(self):
        return {'movie' : ['release_year'],
                'writer': None,
                'director': None
                }

    def get_wide_cols(self):
        return {'movie' : ['has_genre', 'has_tags', 'in_language', 'starred_actors', 'type'],
                'writer': ['has_genre', 'type'],
                'director': ['has_genre', 'type']
                }







