<h1 align="center">
    SLiCE
</h1>
<h4 align="center">Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks</h4>

### Dataset details:
- We use four public benchmark datasets covering multiple applications: e-commerce (Amazon), academic graph
(DBLP), knowledge graphs (Freebase) and social networks (Twitter). Amazon and Twitter data came from https://github.com/THUDM/GATNE. Freebase data came from https://github.com/malllabiisc/CompGCN. DBLP data came from https://github.com/Jhy1993/HAN.
- We introduce
a new knowledge graph from the publicly available real-world Medical Information Mart for Intensive Care III (MIMIC III) dataset
in healthcare domain. https://mimic.physionet.org/
- We also introduce a new knowledge graph from the publicly available Intrusion detection evalution dataset (ISCXIDS2012) https://www.unb.ca/cic/datasets/ids.html
- Note: Relationship IDs have been converted to 1-based indexing if they were previously 0-based

### Install instructions:
- Dependencies: Python 3.6, PyTorch 1.4.0 w/ CUDA 9.2, Pytorch Geometric
- The specific Pytorch Geometric wheels we use are included in the repo for convenience in the 'wheels' directory
```shell
conda create -n slice python=3.6
conda activate slice
pip install -r requirements.txt
```

### Training:
```shell
python main.py \
    --data_name 'amazon_s' \
    --data_path 'data' \
    --outdir 'output/amazon_s' \
    --pretrained_embeddings 'data/amazon_s/amazon_s.emd' \
    --n_epochs 10 \
    --n_layers 4 \
    --n_heads 4 \
    --gcn_option 'no_gcn' \
    --node_edge_composition_func 'mult' \
    --ft_input_option 'last4_cat' \
    --path_option 'shortest' \
    --ft_n_epochs 10 \
    --num_walks_per_node 1 \
    --max_length 6 \
    --walk_type 'dfs' \
    --is_pre_trained
```

### Citation:
Please cite the following paper if you use this code in your work.
```bibtex
@inproceedings{wang2020self,
  title={Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks},
  author={Wang, Ping and Agarwal, Khushbu and Ham, Colby and Choudhury, Sutanay and Reddy, Chandan K},
  booktitle={Proceedings of The Web Conference 2021},
  year={2021}
}
```
