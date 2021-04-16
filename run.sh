#!/bin/bash

#SBATCH -n 1
#SBATCH -A ST_GRAPHS
#SBATCH -t 23:59:00
##SBATCH -p shared_dlt
#SBATCH -p dl
#SBATCH --gres=gpu:1


#module purge
#module load cuda/9.2.148 
#module load python/anaconda3.2019.3
#module load gcc/5.2.0
#source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh
#source activate slice


REPO_DIR=~/Documents/deepcare/SLiCE/SLiCE


n_heads=2 
n_layers=2 
n_pretrain_epochs=2
ft_n_epochs=2
walk_type='dfs'
max_length=6
is_pre_trained=True
gcn_option=no_gcn 
ft_input_option='last4_cat'
path_option='random'
#dataset='amazon_s'
dataset='dblp'
#dataset=''

#data_path=/projects/streaming_graph/contextual_embeddings/datasets/$dataset/$walk_type\_w$n_walks_per_node\_l$max_length/
data_path="${REPO_DIR}/data"
node_edge_composition_func=mult
#pretrained_embeddings=$data_path/../act_$dataset\_mult_500.out
#pretrained_method='compgcn'
#pretrained_embeddings=/projects/streaming_graph/contextual_embeddings/datasets/$dataset/$dataset\.emd
pretrained_embeddings="${data_path}/${dataset}/${dataset}.emd"
#outdir=$data_path/$dataset/results\_layer$n_layers\_nh$n_heads\_wpn$n_walks_per_node\_$walk_type\_l$max_length\_$is_pretrain\_$gcn_option\_$path_option
outdir="${REPO_DIR}/output/${dataset}"
mkdir $outdir

#log_file=$results/$dataset/$dataset\_wkfl$wkfl\_layer$n_layers\_nh$n_heads\_wpn$n_walks_per_node\_$walk_type\_l$max_length\_$is_pretrain\_$gcn_option\_$path_option\.log
#rm $log_file

python -m pdb -c continue main.py \
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
    --ft_n_epochs $ft_n_epochs \
    --max_length $max_length \
    --walk_type $walk_type \
    --is_pre_trained $is_pre_trained
    #>> $log_file
