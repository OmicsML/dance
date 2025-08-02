#!/bin/sh --login
# This is an optional installation script for setting up the DANCE package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch, Pytorch Geometric (PYG), and Deep Graph Library (DGL).
#
# Example:
# $ source install.sh cu118  # install dance with CUDA 11.8
#
# To uninstall and remove the dance environment:
# $ conda remove -n dance --all

trap "echo Try using source instead of sh? && trap - ERR && return 1" ERR

# Check required version specification input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu118,cu121]"
    return 1
fi

# Check optional conda env name input
if [ -z $2 ]; then
    envname=dance
    echo Conda environment name not specified, using the default option: dance
else
    envname=$2
    echo Conda environment set to: ${envname}
fi

# Torch related dependency versions
PYTORCH_VERSION=2.1.1  # XXX: pytorch>=2.1.2 incompatibility issue with DGL 1.1.3 (https://discuss.dgl.ai/t/4244)
TORCHVISION_VERSION=0.16.1
PYG_VERSION=2.4.0
DGL_VERSION=1.1.3  # XXX: 2.0.0 issues with GLIBC (https://github.com/dmlc/dgl/issues/7046)

# Set CUDA variable (use CPU if not set)
CUDA_VERSION=${1:-cpu}
echo "CUDA_VERSION=${CUDA_VERSION}"
case $CUDA_VERSION in
    cpu)
        PYTORCH_CHANNEL="--index-url https://download.pytorch.org/whl/cpu"
        DGL_CHANNEL="-f https://data.dgl.ai/wheels/repo.html"
        ;;
    cu118 | cu121)
        PYTORCH_CHANNEL="--index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
        DGL_CHANNEL="-f https://data.dgl.ai/wheels/${CUDA_VERSION}/repo.html"
        ;;
    *)
        echo "ERROR: Unrecognized CUDA_VERSION=${CUDA_VERSION}"
        return 1
        ;;
esac

# Create environment
conda create -n ${envname} python=3.11 -y
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ${envname}

# Install CUDA enabled dependencies
pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} ${PYTORCH_CHANNEL}
pip install torch_geometric==${PYG_VERSION}
pip install dgl==${DGL_VERSION} ${DGL_CHANNEL}

# Install the rest of the dependencies
pip install -r requirements.txt

# Finally, install the DANCE pckage
pip install -e .

printf "Successfully installed DANCE, be sure to activate the conda environment via:\n"
printf "\n    \$ conda activate ${envname}\n"
printf "\nTo uninstall and remove the DANCE environment:\n"
printf "\n    \$ conda remove -n ${envname} --all\n\n"

trap - ERR
