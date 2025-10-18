#!/bin/bash

# generates python env on orin nano with torch and torchvision wheels
# tested on jetpack 6.2.1 with python 3.10.13 and cuda 12.6

# base setup from original repo


echo "Creating venv"
python -m venv .venv
source .venv/bin/activate

echo "Installing packages from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

# graph neural network (build from source)
export TORCH_CUDA_ARCH_LIST="8.7"
export FORCE_CUDA=1
echo "Building from source torch graph neural network packages, this may take a while..."
pip install --no-build-isolation --use-pep517 torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# system dependencies
echo "Installing system dependencies: libyaml-cpp-dev cudss"
sudo apt-get update
sudo apt-get -y install libyaml-cpp-dev

apt list --installed | grep cudss
HAS_CUDSS=$?

if [ $HAS_CUDSS -ne 0 ]; then
    wget https://developer.download.nvidia.com/compute/cudss/0.6.0/local_installers/cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
    sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
    sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.6.0/cudss-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cudss
    rm cudss-local-tegra-repo-ubuntu2204-0.6.0_0.6.0-1_arm64.deb
fi

# test
echo "Testing torch and torchvision installation (gpu detection)..."
python -c "import torch; import torchvision; print(torch.__version__); print(torchvision.__version__); print('detected gpu' if torch.cuda.is_available() else 'no gpu detected')" | echo "torch and torchvision installed"
