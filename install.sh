#!/bin/sh --login
# This is an optional installation script for setting up the pydance package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch, Pytorch Geometric (PYG), and Deep Graph Library (DGL).
#
# Example:
# $ source install.sh cu102  # install pydance with CUDA 10.2
#
# To uninstall and remove the pydance environment:
# $ conda remove -n pydance --all

# Check input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu102,cu113]"
    return 1
fi

# Torch related dependency versions
PYTORCH_VERSION=1.11.0
PYG_VERSION=2.0.4
DGL_VERSION=0.9

# Set CUDA variable (use CPU if not set)
CUDA_VERSION=${1:-cpu}
echo "CUDA_VERSION=${CUDA_VERSION}"
case $CUDA_VERSION in
    cpu)
        TORCH_OPT="cpuonly"
        DGL_OPT="dgl"
        ;;
    cu102)
        TORCH_OPT="cudatoolkit=10.2"
        DGL_OPT="dgl-cuda10.2"
        ;;
    cu113)
        TORCH_OPT="cudatoolkit=11.3"
        DGL_OPT="dgl-cuda11.3"
        ;;
    *)
        echo "ERROR: Unrecognized CUDA_VERSION=${CUDA_VERSION}"
        return 1
        ;;
esac

# Create environment
conda create -n pydance python=3.8 -y
conda activate pydance

# Install CUDA enabled dependencies
conda install pytorch=${PYTORCH_VERSION} torchvision ${TORCH_OPT} -c pytorch -y
conda install ${DGL_OPT} -c dglteam -y
pip install torch-geometric==${PYG_VERSION} torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html

# Finally, install pydance
pip install -e .

printf "Successfully installed pydance, be sure to activate the pydance conda environment via:\n"
printf "\n    \$ conda activate pydance\n"
printf "\nTo uninstall and remove the pydance environment:\n"
printf "\n    \$ conda remove -n pydance --all\n\n"
