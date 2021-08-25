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
    --data_name $dataset \
    --data_path $data_path \
    --outdir $outdir \
    --pretrained_embeddings $pretrained_embeddings \
    --n_epochs $n_pretrain_epochs \
    --n_layers $n_layers \
    --n_heads $n_heads \
    --gcn_option $gcn_option \
    --node_edge_composition_func $node_edge_composition_func \
    --ft_input_option $ft_input_option \
    --path_option $path_option \
    --ft_n_epochs $n_ft_epochs \
    --num_walks_per_node $n_walks_per_node \
    --max_length $max_length \
    --walk_type $walk_type \
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
