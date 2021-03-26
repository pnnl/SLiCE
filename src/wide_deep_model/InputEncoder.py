import collections
import pprint
import re

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

class NodeEncoder(torch.nn.Module):
    
    def __init__(self, base_embedding_dim, num_nodes,
        pretrained_node_embedding_tensor, is_pre_trained):
        
        super().__init__()
        self.pretrained_node_embedding_tensor = pretrained_node_embedding_tensor
        self.base_embedding_dim = base_embedding_dim

        if (is_pre_trained == False):
            self.base_embedding_layer = \
                torch.nn.Embedding(num_nodes, base_embedding_dim).cuda()
            self.base_embedding_layer.weight.data.uniform_(-1, 1)
        else:
            # print("NodeEncoder: Node2vec embedding size", 
            #     pretrained_node_embedding_tensor.size())
            self.base_embedding_layer = \
                torch.nn.Embedding.from_pretrained(pretrained_node_embedding_tensor).cuda()

    def forward(self, node_id):
        node_id = torch.LongTensor([node_id]).cuda()
        x_base = self.base_embedding_layer(node_id)

        return x_base