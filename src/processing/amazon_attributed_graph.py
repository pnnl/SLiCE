
#!/usr/bin/python
import io
import os
import json
import networkx as nx
import pandas as pd
from collections import Counter
from attributed_graph import AttributedGraph

class AmazonGraph(AttributedGraph):
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

        #get product attributes
        self.product_attr = self.load_product_attributes(main_dir + "/metadata/")
        # load user-product graph and infer user attributes from product
        # attributes
        self.user_attr = self.load_graph(main_dir + "/5core/")
        self.create_normalized_node_id_map()
        self.node_attr_dfs = {
            'user': pd.DataFrame.from_dict(self.user_attr, orient='index'),
            'product': pd.DataFrame.from_dict(
                self.product_attr, orient='index'
            )
        }
        self.get_nodeid2rowid()
        self.get_rowid2nodeid()
        self.get_rowid2vocabid_map()
        self.set_relation_id()
        self.false_edge_gen = false_edge_gen

    def load_graph(self, main_dir):
        """ 
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.node_attr_dfs
        3) self.unique_relations
        4) self.node_types
        
        Amazon Graph Node Types: [user, Industrial and Scientific products, Office
        Products, Video Games]

        Amazon Node Attributes:
        User: None
        Industrial/Office/Games: price, rank, category, brand 
        Note: Graphs contains amazon original graph, filtered to
        1) Filter products that have at least 5 reviews
        2) Filter users that have at least written reviews for 5 different
        products (there were some cases where a reviewer reviewed the same
                      item multiple times)
        """
        self.user_product_categories = dict()
        for filename in os.listdir(main_dir):
            if filename.endswith(".json"):
                user_product_reviews = \
                    json.load(io.open(main_dir + filename, "r",
                                      encoding='utf8'))
                for row_id, review_data in user_product_reviews.items():
                    user_id = review_data['reviewerID']
                    product_id = review_data['asin']
                    rating = review_data['overall']
                    self.G.add_edge(user_id, product_id, label=rating)
                    #self.G.add_edge(product_id, user_id, label=rating)
                    self.unique_relations.add(rating)
                    self.node_types[user_id] = 'user'
                    self.node_types[product_id] = 'product'
                    product_type = filename[0:filename.find('.json')]
                    #self.node_types[product_id] = product_type
                    self.update_user_product_category(
                        self.user_product_categories, user_id, product_id)
        user_attributes = \
            self.get_topk_user_product_categories(self.user_product_categories)
        return user_attributes

    def update_user_product_category(self, user_product_categories, user_id,
                                     product_id):
        if product_id in self.product_attr:
            product_categories = \
                self.product_attr[product_id]['all_categories']
            if user_id not in user_product_categories:
                user_product_categories[user_id] = list()
            for x in product_categories:
                user_product_categories[user_id].append(x)
        return

    def get_topk_user_product_categories(self, user_product_categories, k=5):
        """
        Input is a dictionary of {user_id : list[product_types]}
        output is {user_id -> list[product_types_topk]}
        """
        user_attr = dict()
        for user, product_categories in user_product_categories.items():
            user_top_categories = sorted(Counter(product_categories).items(),
                                         key=lambda kv: kv[1])
            user_attr[user] = dict()
            user_attr[user]['category'] = user_top_categories[-k:][-1][0]
            user_attr[user]['type'] = user
        return user_attr

        
    def load_product_attributes(self, metadata_dir):
        product_attr = dict()
        for filename in os.listdir(metadata_dir):
            if filename.endswith(".json"):
                product_metadata = \
                    json.load(
                        io.open(
                            metadata_dir + filename, "r", encoding='utf8'
                        )
                    )
                for row_id, product_data in product_metadata.items():
                    try:
                        product_id = product_data['asin']
                        product_category = product_data['category']
                        brand = product_data['brand']
                        price = product_data['price']
                        rank = product_data['rank']
                        if isinstance(rank, list):
                            rank = ' '.join(rank)
                        main_category = product_data['main_cat']
                        try:
                            int_rank_str = \
                                rank[rank.find('#') + 1: rank.find('in')]
                            int_rank_str = \
                                int_rank_str.replace(",", '').strip()
                            int_rank = int(int_rank_str)
                        except ValueError:
                            int_rank = -1

                        try:
                            float_price_str = price[price.find('$') + 1:]
                            float_price_str = float_price_str.strip()
                            float_price = float(float_price_str)
                        except ValueError:
                            float_price = -1

                        product_attr[product_id] = {
                            'all_categories': product_category,
                            'brand': brand,
                            'price': float_price,
                            'rank': int_rank,
                            'category': main_category,
                            'type': 'product'
                        }
                        also_buy = product_data['also_buy']
                        if also_buy is not None:
                            for x in also_buy:
                                self.G.add_edge(
                                    product_id, x, label='also_buy')
                    except KeyError:
                        print("Incomplete data for product, ignoring : ",
                              product_id)
        self.unique_relations.add('also_buy')
        self.clean_up_also_buy_products(product_attr)
        return product_attr

    def clean_up_also_buy_products(self, product_attr):
        new_graph = nx.Graph()
        for u, v in self.G.edges():
            edgelabel = self.G.get_edge_data(u, v)['label']
            if(edgelabel == 'also_buy') and (u not in product_attr  or
                                             v not in product_attr):
                continue
            else:
                new_graph.add_edge(u,v, label=edgelabel)
        self.G = new_graph

    def get_continuous_cols(self):
        return {'user' : None,
                'product': ['rank', 'price'],
                }

    def get_wide_cols(self):
                {'user': ['category', 'type'],
                'product' : ['brand', 'category', 'type']
                }







