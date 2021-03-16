import argparse
import glob
import json
import os
import pickle
import re
import shutil
import sys
import time
from typing import List

import networkx as nx
import pandas as pd
import torch
import torch.nn

from src.processing.attributed_graph import AttributedGraph
from src.processing.context_generator import ContextGenerator
from src.processing.dblp_attributed_graph import DBLPGraph
from src.processing.freebase_attributed_graph import FreebaseGraph
from src.processing.generic_attributed_graph import GenericGraph
from src.utils.data_utils import (CreateFinetuneBatches,
                                  CreateFinetuneBatches2, create_batches,
                                  load_pretrained_node2vec, split_data)
from src.utils.evaluation import run_evaluation_graph, run_evaluation_kdd_2019
from src.utils.utils import load_pickle, show_progress


class LinkPrediction(torch.nn.Module):
    """
    Implementation of link prediction strategies based on src
    and dst node embeddings. Ranges from dot product based implementation
    to TransE like approach.
    """

    def __init__(self, opt, ent2id, rel2id, pretrained_node_embeddings):
        """
        Args:
            opt: specify the link prediction strategy.
        """
        super(LinkPrediction, self).__init__()
        self.opt = opt
        self.ent2id = ent2id
        self.pretrained_node_embeddings = pretrained_node_embeddings

    def __predict_via_dot_product(self, edge):

        source_id = self.ent2id[edge[1]]
        target_id = self.ent2id[edge[2]]
        source_vec = self.pretrained_node_embeddings[source_id]
        target_vec = self.pretrained_node_embeddings[target_id]
        source_vec = source_vec.unsqueeze(0).unsqueeze(0)
        target_vec = target_vec.unsqueeze(0).unsqueeze(0).transpose(1, 2)
        score = torch.bmm(source_vec, target_vec)
        score = torch.sigmoid(score).data.cpu().numpy().tolist()[0][0][0]

        return score

    def forward(self, triple):
        if self.opt == 'dot_product':
            score = self.__predict_via_dot_product(triple)

        return score


def link_prediction_eval(test_edges, ent2id, rel2id, pretrained_node_embeddings):
    """
    Args:
        test_edges (List[triple]): list of edges loaded from test.txt
        node_embedding (NodeEmbedding): class implementing NodeEmbedding 
                interface

    Returns:
        Precision, Recall, F-Score for link prediction test
    """
    link_prediction = \
        LinkPrediction('dot_product', ent2id, rel2id,
                       pretrained_node_embeddings)

    pred_data = []
    true_data = []
    true_num = 0
    for edge in test_edges:
        label = edge[3]
        if label == '0':
            true_data.append(0)
        elif label == '1':
            true_data.append(1)
            true_num += 1
        else:
            print(f"Unhandled label value: {label}, exiting...")
            sys.exit(1)
        score = link_prediction(edge)
        pred_data.append(score)

    run_evaluation_kdd_2019(pred_data, true_data, true_num)


def get_test_edges(paths: List[str], sep: str):
    edges = set()
    for path in paths:
        with open(path, 'r') as f:
            for l in f:
                tokens = l.strip().split(sep)
                etype = tokens[0]
                source = tokens[1]
                destination = tokens[2]
                label = tokens[3]
                edge = (etype, source, destination, label)
                edges.add(edge)

    return edges


def get_id_map(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_sep(data_name: str):
    sep = ' '
    #if data_name == 'freebase':
    #    sep = ' '
    #elif data_name == 'amazon':
    #    sep = ' '
    return sep


def main(args):
    stime = time.time()

    # Load pretrained embedding from CompGCN
    pretrained_node_embedding = load_pickle(args.pretrained_embeddings)
    print('No. of nodes with pretrained embedding from CompGCN: ',
          len(pretrained_node_embedding))

    test_path = f"{args.data_path}/compgcn_output/{args.data_name}/test.txt"
    valid_path = f"{args.data_path}/compgcn_output/{args.data_name}/valid.txt"
    test_edges_paths = [test_path, valid_path]
    sep = get_sep(args.data_name)
    test_edges = list(get_test_edges(test_edges_paths, sep))
    print('No. edges in test data: ', len(test_edges))

    id_maps_dir = f"{args.data_path}/compgcn_output/{args.data_name}"
    ent2id = get_id_map(f"{id_maps_dir}/ent2id.txt")
    rel2id = get_id_map(f"{id_maps_dir}/rel2id.txt")
    print(len(ent2id), len(rel2id))

    # Run link predication with the embedding from CompGCN
    link_prediction_eval(
        test_edges, ent2id, rel2id, pretrained_node_embedding)

    etime = time.time()
    elapsed = etime - stime
    print(f"running time(seconds) on {args.data_name} data: {elapsed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--etask', default='train',
                        help='running task, train/test')
    parser.add_argument('--data_name', default='dblp', help='name of the dataset')
    parser.add_argument('--data_path', default='../data_kdd20',
                        help='path to dataset')
    parser.add_argument('--outdir', default='test_out',
                        help='path to output dir')
    parser.add_argument('--pretrained_embeddings')
    # process walks
    parser.add_argument('--beam_width', default=2,
                        help='beam width used for generating random walks')
    parser.add_argument('--num_walks_per_node',
                        default=1, help='walks per node')
    parser.add_argument('--walk_type', default='dfs', help='walk type bfs/dfs')
    parser.add_argument('--max_length', default=6, help='max length of walks')
    parser.add_argument('--n_pred', default=1,
                        help='number of tokens masked to be predicted')
    parser.add_argument('--max_pred', default=1,
                        help='max number of tokens masked to be predicted')
    # pre-training
    parser.add_argument('--node_edge_composition_func', default='mult', help='mult|no_rel|sub|corr')
    parser.add_argument('--emb_dir')

    args = parser.parse_args()

    if not args.pretrained_embeddings:
        args.pretrained_embeddings = \
            f"{args.emb_dir}/{args.data_name}/act_{args.data_name}_{args.node_edge_composition_func}_500.out"

    print("Args ", str(args))
    main(args)
