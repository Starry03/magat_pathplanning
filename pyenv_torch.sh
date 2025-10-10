#!/bin/bash

# generates python env on orin nano with torch and torchvision wheels
# tested on jetpack 6.2.1 with python 3.10.13 and cuda 12.6

# base setup from original repo

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt

# fork adaptation for jetson

# torch
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/590/92ab729aee2b8/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=59092ab729aee2b8937d80cc1b35d1128275bd02a7e1bc911e7efa375bd97226

# torchvision
pip install https://pypi.jetson-ai-lab.io/jp6/cu126/+f/1c0/3de08a69e9554/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=1c03de08a69e95542024477e0cde95fab3436804917133d3f00e67629d3fe902

# graph neural network (build from source)
export TORCH_CUDA_ARCH_LIST="8.7"
export FORCE_CUDA=1
pip install --no-build-isolation --use-pep517 torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# cudss
if [ ! -f cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb ]; then
    wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
    sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
    sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cudss
fi

# missing dependencies ignored by requirements.txt :)
python3 -m pip install opencv-python pyyaml hashids drawsvg==1.9.0 seaborn torchinfo tensorflow
sudo apt-get update
sudo apt-get -y install libyaml-cpp-dev

# test
python -c "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__); print('detected gpu' if torch.cuda.is_available() else 'no gpu detected')" | echo "torch and torchvision installed"
