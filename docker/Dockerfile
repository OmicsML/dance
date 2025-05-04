# basic image
FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

# install Python3 and pip3
RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get install -y wget

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/miniconda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

RUN conda init bash

# install PyTorch and torchvision
# RUN conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# copy all files under current folder to container
COPY . /workspace

# set working directory
WORKDIR /workspace

# running command line
RUN /bin/bash -c "source install.sh cu113 dance-env"
