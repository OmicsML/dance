#!/bin/sh --login
# This is an optional installation script for setting up the pydance package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch, Pytorch Geometric (PYG), and Deep Graph Library (DGL).
#
# Example:
# $ source install.sh cu102  # install pydance with CUDA 10.2
#
# To uninstall and remove the dance environment:
# $ conda remove -n dance --all

trap "echo Try using source instead of sh? && trap - ERR && return 1" ERR

# Check required version specificiation input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu102,cu113]"
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
PYTORCH_VERSION=1.12.1
PYG_VERSION=2.1.0
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
conda create -n ${envname} python=3.8 -y
conda activate ${envname}

# Install CUDA enabled dependencies
conda install pytorch=${PYTORCH_VERSION} torchvision ${TORCH_OPT} -c pytorch -y
conda install ${DGL_OPT} -c dglteam -y
pip install torch-geometric==${PYG_VERSION} torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VERSION}.html

# Finally, install pydance
pip install -e .

printf "Successfully installed pydance, be sure to activate the conda environment via:\n"
printf "\n    \$ conda activate ${envname}\n"
printf "\nTo uninstall and remove the pydance environment:\n"
printf "\n    \$ conda remove -n ${envname} --all\n\n"

trap - ERR
