#!/bin/sh --login
# This is an optional installation script for setting up the DANCE package
# without much headache of dealing with setting up CUDA enabled packages, such
# as PyTorch, Pytorch Geometric (PYG), and Deep Graph Library (DGL).
#
# Example:
# $ source install.sh cu117  # install dance with CUDA 11.7
#
# To uninstall and remove the dance environment:
# $ conda remove -n dance --all

trap "echo Try using source instead of sh? && trap - ERR && return 1" ERR

# Check required version specification input
if [ -z $1 ]; then
    echo "ERROR: Please provide CUDA information, available options are [cpu,cu117]"
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
PYTORCH_VERSION=2.0.0
TORCHVISION_VERSION=0.15.0
TORCHAUDIO_VERSION=2.0.0
PYG_VERSION=2.3.0
DGL_VERSION=1.0.1

# Set CUDA variable (use CPU if not set)
CUDA_VERSION=${1:-cpu}
echo "CUDA_VERSION=${CUDA_VERSION}"
case $CUDA_VERSION in
    cpu)
        PYTORCH_CUDA_OPT="cpuonly -c pytorch"
        DGL_CHANNEL="dglteam"
        ;;
    cu117)
        PYTORCH_CUDA_OPT="pytorch-cuda=11.7 -c pytorch -c nvidia"
        DGL_CHANNEL="dglteam/label/cu117"
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
conda install pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} \
    torchaudio=${TORCHAUDIO_VERSION} ${PYTORCH_CUDA_OPT} -y
conda install pyg=${PYG_VERSION} -c pyg -y
conda install dgl=${DGL_VERSION} -c ${DGL_CHANNEL} -y

# Install the rest of the dependencies
pip install -r requirements.txt

# Finally, install the DANCE pckage
pip install -e .

printf "Successfully installed DANCE, be sure to activate the conda environment via:\n"
printf "\n    \$ conda activate ${envname}\n"
printf "\nTo uninstall and remove the DANCE environment:\n"
printf "\n    \$ conda remove -n ${envname} --all\n\n"

trap - ERR
