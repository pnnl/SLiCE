#!/usr/bin/bash

#SBATCH --job-name=slice
#SBATCH --output=logs/%x-%j.out
#SBATCH -A ST_GRAPHS
#SBATCH -p dl
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 3-23:59:00

module purge
module load cuda/9.2.148 
module load python/anaconda3.2019.3
module load gcc/5.2.0

source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh
source activate slice

# debug info
script_name='./slice.sh'
script_name_during_run=${0}
echo -e "\n\n"
echo "script_name: "${script_name}
echo "script_name_during_run: "${script_name_during_run}

# script parameters

home_dir=$HOME
repo_dir="{$home_dir}/SLiCE"
data_name='amazon_s'
data_path="{$repo_dir}/data"
outdir="{$repo_dir}/output/{$data_name}"
pretrained_embeddings="{$repo_dir}/data/{$data_name}/{$data_name}.emd"
n_epochs=2
n_layers=2
n_heads=2
ft_n_epochs=2
max_length=6
gcn_option="no_gcn"
node_edge_composition_func="mult"
ft_input_option="last4_cat"
path_option="random"
walk_type="dfs"

echo "home_dir: "${home_dir}
echo "repo_dir: "${repo_dir}
echo "data_name: "$data_name
echo "data_path: "$data_path
echo "outdir: "$outdir
echo "pretrained_embeddings: "$pretrained_embeddings
echo "n_epochs: "$n_epochs
echo "n_layers: "$n_layers
echo "n_heads: "$n_heads
echo "ft_n_epochs: "$ft_n_epochs
echo "max_length: "$max_length
echo "gcn_option: "$gcn_option
echo "node_edge_composition_func: "$node_edge_composition_func
echo "ft_input_option: "$ft_input_option
echo "path_option: "$path_option
echo "walk_type: "$walk_type

cd $repo_dir

python main.py \
    --data_name $data_name \
    --data_path $data_path \
    --outdir $outdir \
    --pretrained_embeddings $pretrained_embeddings \
    --n_epochs $n_epochs \
    --n_layers $n_layers \
    --n_heads $n_heads \
    --gcn_option $gcn_option \
    --node_edge_composition_func $node_edge_composition_func \
    --ft_input_option $ft_input_option \
    --path_option $path_option \
    --ft_n_epochs $ft_n_epochs \
    --max_length $max_length \
    --walk_type $walk_type \
    --is_pre_trained