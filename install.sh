# This install script is specific to Linux
# These modules may have slightly different versions depending on your hpc setup
# The main important thing is to have your environment know the 'conda' command
# and also having CUDA 9.2 specified for the system
module purge
module load cuda/9.2.148 
module load python/anaconda3.2019.3
module load gcc/5.2.0
source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh

echo "Creating conda environment"
conda create -n slice python=3.6 -y
conda activate slice

echo "Installing pytorch-geometric local wheels"
pip install wheels/linux/torch_cluster-1.5.4-cp36-cp36m-linux_x86_64.whl
pip install wheels/linux/torch_scatter-2.0.4-cp36-cp36m-linux_x86_64.whl
pip install wheels/linux/torch_sparse-0.6.1-cp36-cp36m-linux_x86_64.whl
pip install wheels/linux/torch_spline_conv-1.2.0-cp36-cp36m-linux_x86_64.whl

echo "Installing pytorch and torchvision"
# Explicit links if the command below doesn't work
# https://download.pytorch.org/whl/cu92/torchvision-0.5.0%2Bcu92-cp36-cp36m-linux_x86_64.whl
# https://download.pytorch.org/whl/cu92/torch-1.4.0%2Bcu92-cp36-cp36m-linux_x86_64.whl
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

echo "Install requirments.txt"
pip install -r requirements.txt

echo "Done"