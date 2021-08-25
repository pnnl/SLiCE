conda create -n slice python=3.6
conda activate slice
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install ./wheels/linux/torch_cluster-1.5.4-cp36-cp36m-linux_x86_64.whl
pip install ./wheels/linux/torch_scatter-2.0.4-cp36-cp36m-linux_x86_64.whl
pip install ./wheels/linux/torch_sparse-0.6.1-cp36-cp36m-linux_x86_64.whl
pip install ./wheels/linux/torch_spline_conv-1.2.0-cp36-cp36m-linux_x86_64.whl
pip install -r requirements.txt