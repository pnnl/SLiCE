#!/usr/bin/bash

#SBATCH --job-name=slice-amazon_s
#SBATCH --output=logs/%x-%j.out
#SBATCH -A ST_GRAPHS
#SBATCH -p dl
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 3-23:59:00

module purge
module load cuda/9.2.148 
module load python/anaconda3.2019.3

source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh
source activate slice
date

# debug info
script_name='./slice.sh'
script_name_during_run=${0}
echo -e "\n\n"
echo "script_name: "${script_name}
echo "script_name_during_run: "${script_name_during_run}

# script parameters
home_dir=$HOME
repo_dir="${home_dir}/SLiCE"

# For non-cyber datasets
data_name='amazon_s'
data_path="${repo_dir}/data"
outdir="${repo_dir}/output/${data_name}"
pretrained_embeddings="${repo_dir}/data/${data_name}/${data_name}.emd"
pretrained_method='node2vec'

# For cyber datasets
# data_name='cyber_17'
# data_path="${repo_dir}/data/cyber/processed"
# outdir="${repo_dir}/output/${data_name}"
# pretrained_embeddings="${data_path}/${data_name}/${data_name}.emd"

# script parameters
batch_size=128
ft_batch_size=128
lr=0.0001
ft_lr=0.001
n_epochs=10
ft_n_epochs=10
n_heads=4
n_layers=4
beam_width=2
walk_type='dfs'
max_length=6
gcn_option='no_gcn'
node_edge_composition_func='mult'
ft_input_option='last4_cat'
path_option='shortest'

echo "home_dir: "$home_dir
echo "repo_dir: "$repo_dir
echo "data_name: "$data_name
echo "data_path: "$data_path
echo "outdir: "$outdir
echo "pretrained_embeddings: "$pretrained_embeddings
echo "pretrained_method: "$pretrained_method
echo "batch_size: "$batch_size
echo "ft_batch_size: "$ft_batch_size
echo "lr: "$lr
echo "ft_lr: "$ft_lr
echo "n_epochs: "$n_epochs
echo "ft_n_epochs: "$ft_n_epochs
echo "n_heads: "$n_heads
echo "n_layers: "$n_layers
echo "beam_width: "$beam_width
echo "walk_type: "$walk_type
echo "max_length: "$max_length
echo "gcn_option: "$gcn_option
echo "node_edge_composition_func: "$node_edge_composition_func
echo "ft_input_option: "$ft_input_option
echo "path_option: "$path_option

cd $repo_dir

python main.py \
    --data_name $data_name \
    --data_path $data_path \
    --outdir $outdir \
    --pretrained_embeddings $pretrained_embeddings \
    --pretrained_method $pretrained_method \
    --batch_size $batch_size \
    --ft_batch_size $ft_batch_size \
    --lr $lr \
    --ft_lr $ft_lr \
    --n_epochs $n_epochs \
    --ft_n_epochs $ft_n_epochs \
    --n_heads $n_heads \
    --n_layers $n_layers \
    --beam_width $beam_width \
    --walk_type $walk_type \
    --max_length $max_length \
    --gcn_option $gcn_option \
    --node_edge_composition_func $node_edge_composition_func \
    --ft_input_option $ft_input_option \
    --path_option $path_option \
    --is_pre_trained

echo "DONE"
date