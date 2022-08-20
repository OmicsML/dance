<p align="center">
  <img
       src="https://github.com/OmicsML/dance/blob/main/imgs/dance_logo.jpg"
       style="width:100%; height:100%; object-fit:cover;"
  />
</p>

______________________________________________________________________

[![PyPI version](https://badge.fury.io/py/pydance.svg)](https://badge.fury.io/py/pydance)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Documentation Status](https://readthedocs.org/projects/pydance/badge/?version=latest)](https://pydance.readthedocs.io/en/latest/?badge=latest)
[![Test Examples](https://github.com/OmicsML/dance/actions/workflows/test_examples.yml/badge.svg)](https://github.com/OmicsML/dance/actions/workflows/test_examples.yml)

DANCE is a Python toolkit to support deep learning models for analyzing single-cell gene expression at scale. It includes three modules at present:

1. **Single-modality analysis**
1. **Single-cell multimodal omics**
1. **Spatially resolved transcriptomics**

Our goal is to build up a deep learning community for single cell analysis and provide GNN based architecture for users for further development in single cell analysis.

## Usage

### Overview

In release 1.0, the main usage of the PyDANCE is to provide readily available experiment reproduction
(see detail information about the reproduced performance [below](#implemented-algorithms)).
Users can easily reproduce selected experiments presented in the original papers for the computational single-cell methods implemented in PyDANCE, which can be found under [`examples/`](examples).

### Motivation

Computational methods for single-cell analysis are quickly emerging, and the field is revolutionizing the usage of single-cell data to gain biological insights.
A key challenge to continually developing computational single-cell methods that achieve new state-of-the-art performance is reproducing previous benchmarks.
More specifically, different studies prepare their datasets and perform evaluation differently,
and not to mention the compatibility of different methods, as they could be written in different languages or using incompatible library versions.

PyDANCE addresses these challenges by providing a unified Python packge implementing many popular computational single-cell methods (see [Implemented Algorithms](#implemented-algorithms)),
as well as easily reproducible experiments by providing unified tools for

- Data downloading
- Data (pre-)processing and transformation (e.g. graph construction)
- Model training and evaluation

### Example: runing cell-type annotation benchmark using scDeepSort

- Step0. Install PyDANCE (see [Installation](#installation))
- Step1. Navigate to the folder containing the corresponding example scrtip.
  In this case, it is [`examples/single_modality/cell_type_annotation`](examples/single_modality/cell_type_annotation).
- Step2. Obtain command line interface (CLI) options for a particular experiment to reproduce at the end of the
  [script](examples/single_modality/cell_type_annotation/scdeepsort.py).
  For example, the CLI options for reproducing the `Mouse Brain` experiment is
  ```bash
  python scdeepsort.py --data_type scdeepsort --tissue Brain --test_data 2695
  ```
- Step3. Wait for the experiment to finsh and check results.

## Installation

<H3>Quick install</H3>

The full installation process might be a bit tedious and could involve some debugging when using CUDA enabled packages.
Thus, we provide an `install.sh` script that simplifies the installation process, assuming the user have [conda](https://conda.io/projects/conda/en/latest/index.html) set up on their machines.
The installation script creates a conda environment `pydance` and install the PyDANCE package along with all its dependencies with a apseicifc CUDA version.
Currently, three options are accepted: `cpu`, `cu102`, and `cu113`.
For example, to install the DANCE package using CUDA10.2, simply run:

```bash
git clone git@github.com:OmicsML/dance.git
cd dance

source install.sh cu102
```

<details>
<summary><H3>Custom install</H3></summary>
<br>

**Step1. Setup environment**

First create a conda environment for pydance (optional)

```bash
conda create -n pydance python=3.8 -y && conda activate dance-dev
```

Then, install CUDA enabled packages (PyTorch, PyG, DGL) with CUDA 10.2:

```bash
conda install pytorch=1.12.1 torchvision cudatoolkit=10.2 -c pytorch -y
conda install dgl-cu102 -c dglteam -y
pip install torch-geometric==2.1.0 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu102.html
```

Alternatively, install these dependencies for CPU only:

```bash
conda install pytorch=1.12.1 torchvision cpuonly -c pytorch -y
conda install dgl -c dglteam
pip install torch-geometric==2.1.0 torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
```

**Note:** If you installed PyG using conda and encountered an issue with `GLIBC_2.27` when importing `torch_geometric.nn`,
then you may need to uninstall `torch-spline-conv` (see https://github.com/pyg-team/pytorch_geometric/issues/3593)

```bash
pip uninstall torch-spline-conv
```

For more information about installation or other CUDA version options, check out the installation pages for the corresponding packges

- [PyTorch](https://pytorch.org/get-started/)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [DGL](https://www.dgl.ai/pages/start.html)

**Step2. Install PyDANCE**

Install from PyPI

```bash
pip install pydance
```

Install the latest dev version from source

```bash
git clone https://github.com/OmicsML/dance.git
cd dance
pip install -e .
```

</details>

## Implemented Algorithms

**P1** not covered in the first release

### Single Modality Module

#### 1）Imputation

| BackBone            | Model        | Algorithm                                                                                                    | Year | CheckIn |
| ------------------- | ------------ | ------------------------------------------------------------------------------------------------------------ | ---- | ------- |
| GNN                 | GraphSCI     | Imputing Single-cell RNA-seq data by combining Graph Convolution and Autoencoder Neural Networks             | 2021 | ✅       |
| GNN                 | scGNN (2020) | SCGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph                           | 2020 | P1      |
| GNN                 | scGNN (2021) | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses                             | 2021 | ✅       |
| GNN                 | GNNImpute    | An efficient scRNA-seq dropout imputation method using graph attention network                               | 2021 | P1      |
| Graph Diffusion     | MAGIC        | MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data | 2018 | P1      |
| Probabilistic Model | scImpute     | An accurate and robust imputation method scImpute for single-cell RNA-seq data                               | 2018 | P1      |
| GAN                 | scGAIN       | scGAIN: Single Cell RNA-seq Data Imputation using Generative Adversarial Networks                            | 2019 | P1      |
| NN                  | DeepImpute   | DeepImpute: an accurate, fast, and scalable deep neural network method to impute single-cell RNA-seq data    | 2019 | ✅       |
| NN + TF             | Saver-X      | Transfer learning in single-cell transcriptomics improves data denoising and pattern discovery               | 2019 | P1      |

| Model      | Evaluation Metric | Mouse Brain (current/reported) | Mouse Embryo (current/reported) |
| ---------- | ----------------- | ------------------------------ | ------------------------------- |
| DeepImpute | MSE               | 0.12 / N/A                     | 0.12 / N/A                      |
| ScGNN      | MSE               | 0.47 / N/A                     | 1.10 / N/A                      |
| GraphSCI   | MSE               | 0.42 / N/A                     | 0.87 / N/A                      |

Note: the data split modality of DeepImpute is different from ScGNN and GraphSCI, so the results are not comparable.

#### 2）Cell Type Annotation

| BackBone                | Model         | Algorithm                                                                                                    | Year | CheckIn |
| ----------------------- | ------------- | ------------------------------------------------------------------------------------------------------------ | ---- | ------- |
| GNN                     | ScDeepsort    | Single-cell transcriptomics with weighted GNN                                                                | 2021 | ✅       |
| Logistic Regression     | Celltypist    | Automated cell type annotation for scRNA-seq datasets                                                        | 2021 | ✅       |
| Random Forest           | singleCellNet | SingleCellNet: a computational tool to classify single cell RNA-Seq data across platforms and across species | 2019 | ✅       |
| Neural Network          | ACTINN        | ACTINN: automated identification of cell types in single cell RNA sequencing.                                | 2020 | ✅       |
| Hierarchical Clustering | SingleR       | Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.       | 2019 | P1      |
| SVM                     | SVM           | A comparison of automatic cell identification methods for single-cell RNA sequencing data.                   | 2018 | ✅       |

| Model         | Evaluation Metric | Mouse Brain 2695 (current/reported) | Mouse Spleen 1759 (current/reported) | Mouse Kidney 203 (current/reported) |
| ------------- | ----------------- | ----------------------------------- | ------------------------------------ | ----------------------------------- |
| scDeepsort    | ACC               | 0.363/0.363                         | 0.965 /0.965                         | 0.901/0.911                         |
| Celltypist\*  | ACC               | 0.680/0.666                         | 0.966/0.848                          | 0.879/0.832                         |
| singleCellNet | ACC               | 0.693/0.803                         | 0.975/0.975                          | 0.795/0.842                         |
| ACTINN        | ACC               | 0.860/0.778                         | 0.516/0.236                          | 0.829/0.798                         |
| SVM           | ACC               | 0.683/0.683                         | 0.056/0.049                          | 0.704/0.695                         |

Note: * Benchmark datasets were renormalied before running the original implementation of Celltypist to match its form requirements.

#### 3）Clustering

| BackBone    | Model         | Algorithm                                                                                                    | Year | CheckIn |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------ | ---- | ------- |
| GNN         | graph-sc      | GNN-based embedding for clustering scRNA-seq data                                                            | 2022 | ✅       |
| GNN         | scTAG         | ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations                               | 2022 | ✅       |
| GNN         | scDSC         | Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph neural network | 2022 | ✅       |
| GNN         | scGAC         | scGAC: a graph attentional architecture for clustering single-cell RNA-seq data                              | 2022 | P1      |
| AutoEncoder | scDeepCluster | Clustering single-cell RNA-seq data with a model-based deep learning approach                                | 2019 | ✅       |
| AutoEncoder | scDCC         | Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data                   | 2021 | ✅       |
| AutoEncoder | scziDesk      | Deep soft K-means clustering with self-training for single-cell RNA sequence data                            | 2020 | P1      |

| Model         | Evaluation Metric | 10x PBMC (current/reported) | Mouse ES (current/reported) | Worm Neuron (current/reported) | Mouse Bladder (current/reported) |
| ------------- | ----------------- | --------------------------- | --------------------------- | ------------------------------ | -------------------------------- |
| graph-sc      | ARI               | 0.72 / 0.70                 | 0.82 / 0.78                 | 0.57 / 0.46                    | 0.68 / 0.63                      |
| scDCC         | ARI               | 0.82 / 0.81                 | 0.98 / N/A                  | 0.51 / 0.58                    | 0.60 / 0.66                      |
| scDeepCluster | ARI               | 0.81 / 0.78                 | 0.98 / 0.97                 | 0.51 / 0.52                    | 0.56 / 0.58                      |
| scDSC         | ARI               | 0.72 / 0.78                 | 0.84 / N/A                  | 0.46 / 0.65                    | 0.65 / 0.72                      |
| scTAG         | ARI               | 0.75 / N/A                  | 0.96 / N/A                  | 0.53 / N/A                     | 0.60 / N/A                       |

### Multimodality Module

#### 1）Modality Prediction

| BackBone         | Model                    | Algorithm                                                                                          | Year | CheckIn |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN                  | Graph Neural Networks for Multimodal Single-Cell Data Integration                                  | 2022 | ✅       |
| GNN              | ScMoLP                   | Link Prediction Variant of ScMoGCN                                                                 | 2022 | P1      |
| GNN              | scGNN                    | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses                   | 2021 | P1      |
| GNN              | GRAPE                    | Handling Missing Data with Graph Representation Learning                                           | 2020 | P1      |
| Generative Model | SCMM                     | SCMM: MIXTURE-OF-EXPERTS MULTIMODAL DEEP GENERATIVE MODEL FOR SINGLE-CELL MULTIOMICS DATA ANALYSIS | 2021 | ✅       |
| Auto-encoder     | Cross-modal autoencoders | Multi-domain translation between single-cell imaging and sequencing data using autoencoders        | 2021 | ✅       |
| Auto-encoder     | BABEL                    | BABEL enables cross-modality translation between multiomic profiles at single-cell resolution      | 2021 | ✅       |

| Model                    | Evaluation Metric | GEX2ADT (current/reported) | ADT2GEX (current/reported) | GEX2ATAC (current/reported) | ATAC2GEX (current/reported) |
| ------------------------ | ----------------- | -------------------------- | -------------------------- | --------------------------- | --------------------------- |
| ScMoGCN                  | RMSE              | 0.3885 / 0.3885            | 0.3242 / 0.3242            | 0.1778 / 0.1778             | 0.2315 / 0.2315             |
| SCMM                     | RMSE              | 0.6264 / N/A               | 0.4458 / N/A               | 0.2163 / N/A                | 0.3730 / N/A                |
| Cross-modal autoencoders | RMSE              | 0.5725 / N/A               | 0.3585 / N/A               | 0.1917 / N/A                | 0.2551 / N/A                |
| BABEL                    | RMSE              | 0.4335 / N/A               | 0.3673 / N/A               | 0.1816 / N/A                | 0.2394 / N/A                |

#### 2) Modality Matching

| BackBone         | Model                    | Algorithm                                                                                          | Year | CheckIn |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN                  | Graph Neural Networks for Multimodal Single-Cell Data Integration                                  | 2022 | ✅       |
| GNN              | scGNN                    | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses                   | 2021 | P1      |
| Generative Model | SCMM                     | SCMM: MIXTURE-OF-EXPERTS MULTIMODAL DEEP GENERATIVE MODEL FOR SINGLE-CELL MULTIOMICS DATA ANALYSIS | 2021 | ✅       |
| Auto-encoder     | Cross-modal autoencoders | Multi-domain translation between single-cell imaging and sequencing data using autoencoders        | 2021 | ✅       |

| Model                    | Evaluation Metric | GEX2ADT (current/reported) | GEX2ATAC (current/reported) |
| ------------------------ | ----------------- | -------------------------- | --------------------------- |
| ScMoGCN                  | Accuracy          | 0.0827 / 0.0810            | 0.0600 / 0.0630             |
| SCMM                     | Accuracy          | 0.005 / N/A                | 5e-5 / N/A                  |
| Cross-modal autoencoders | Accuracy          | 0.0002 / N/A               | 0.0002 /  N/A               |

#### 3) Joint Embedding

| BackBone         | Model   | Algorithm                                                                                             | Year | CheckIn |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN | Graph Neural Networks for Multimodal Single-Cell Data Integration                                     | 2022 | ✅       |
| Auto-encoder     | scMVAE  | Deep-joint-learning analysis model of single cell transcriptome and open chromatin accessibility data | 2020 | ✅       |
| Auto-encoder     | scDEC   | Simultaneous deep generative modelling and clustering of single-cell genomic data                     | 2021 | ✅       |
| GNN/Auto-ecnoder | GLUE    | Multi-omics single-cell data integration and regulatory inference with graph-linked embedding         | 2021 | P1      |
| Auto-encoder     | DCCA    | Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data             | 2021 | ✅       |

| Model      | Evaluation Metric | GEX2ADT (current/reported) | GEX2ATAC (current/reported) |
| ---------- | ----------------- | -------------------------- | --------------------------- |
| ScMoGCN    | ARI               | 0.706 / N/A                | 0.702 /  N/A                |
| ScMoGCNv2  | ARI               | 0.734 / N/A                | N/A /  N/A                  |
| scMVAE     | ARI               | 0.499 /  N/A               | 0.577 /  N/A                |
| scDEC(JAE) | ARI               | 0.705 /  N/A               | 0.735 /  N/A                |
| DCCA       | ARI               | 0.35 /  N/A                | 0.381 /  N/A                |

#### 4) Multimodal Imputation

| BackBone | Model  | Algorithm                                                                        | Year | CheckIn |
| -------- | ------ | -------------------------------------------------------------------------------- | ---- | ------- |
| GNN      | ScMoLP | Link Prediction Variant of ScMoGCN                                               | 2022 | P1      |
| GNN      | scGNN  | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses | 2021 | P1      |
| GNN      | GRAPE  | Handling Missing Data with Graph Representation Learning                         | 2020 | P1      |

#### 5) Multimodal Integration

| BackBone         | Model    | Algorithm                                                                                                        | Year | CheckIn |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN  | Graph Neural Networks for Multimodal Single-Cell Data Integration                                                | 2022 | P1      |
| GNN              | scGNN    | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses (GCN on Nearest Neighbor graph) | 2021 | P1      |
| Nearest Neighbor | WNN      | Integrated analysis of multimodal single-cell data                                                               | 2021 | P1      |
| GAN              | MAGAN    | MAGAN: Aligning Biological Manifolds                                                                             | 2018 | P1      |
| Auto-encoder     | SCIM     | SCIM: universal single-cell matching with unpaired feature sets                                                  | 2020 | P1      |
| Auto-encoder     | MultiMAP | MultiMAP: Dimensionality Reduction and Integration of Multimodal Data                                            | 2021 | P1      |
| Generative Model | SCMM     | SCMM: MIXTURE-OF-EXPERTS MULTIMODAL DEEP GENERATIVE MODEL FOR SINGLE-CELL MULTIOMICS DATA ANALYSIS               | 2021 | P1      |

### Spatial Module

#### 1）Spatial Domain

| BackBone                         | Model      | Algorithm                                                                                                                                                                     | Year | CheckIn |
| -------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN                              | SpaGCN     | SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network                   | 2021 | ✅       |
| GNN                              | STAGATE    | Deciphering spatial domains from spatially resolved transcriptomics with adaptive graph attention auto-encoder                                                                | 2021 | ✅       |
| Bayesian                         | BayesSpace | Spatial transcriptomics at subspot resolution with BayesSpace                                                                                                                 | 2021 | P1      |
| Pseudo-space-time (PST) Distance | stLearn    | stLearn: integrating spatial location, tissue morphology and gene expression to find cell types, cell-cell interactions and spatial trajectories within undissociated tissues | 2020 | ✅       |
| Heuristic                        | Louvain    | Fast unfolding of community hierarchies in large networks                                                                                                                     | 2008 | ✅       |

| Model   | Evaluation Metric | 151673 (current/reported) | 151676 (current/reported) | 151507 (current/reported) |
| ------- | ----------------- | ------------------------- | ------------------------- | ------------------------- |
| SpaGCN  | ARI               | 0.51  / 0.522             | 0.41 / N/A                | 0.45 / N/A                |
| STAGATE | ARI               | 0.59 / N/A                | 0.60 / 0.60               | 0.608 / N/A               |
| stLearn | ARI               | 0.30 / 0.36               | 0.29 / N/A                | 0.31 / N/A                |
| Louvain | ARI               | 0.31 / 0.33               | 0.2528 / N/A              | 0.28 / N/A                |

#### 2）Cell Type Deconvolution

| BackBone                   | Model        | Algorithm                                                                                                     | Year | CheckIn |
| -------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN                        | DSTG         | DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence                  | 2021 | ✅       |
| logNormReg                 | SpatialDecon | Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data       | 2022 | ✅       |
| NNMFreg                    | SPOTlight    | SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics spots with single-cell transcriptomes | 2021 | ✅       |
| NN Linear + CAR assumption | CARD         | Spatially informed cell-type deconvolution for spatial transcriptomics                                        | 2022 | ✅       |

| Model        | Evaluation Metric | GSE174746 (current/reported) | CARD Synthetic (current/reported) | SPOTlight Synthetic (current/reported) |
| ------------ | ----------------- | ---------------------------- | --------------------------------- | -------------------------------------- |
| DSTG         | MSE               | .172 / N/A                   | .0247 / N/A                       | .042 / N/A                             |
| SpatialDecon | MSE               | .0014 / .009                 | .0077 / N/A                       | .0055 / N/A                            |
| SPOTlight    | MSE               | .0098 / N/A                  | .0246 / 0.118                     | .0109 / .16                            |
| CARD         | MSE               | .0012 / N/A                  | .0078 / 0.0062                    | .0076 / N/A                            |

## Dev notes

### Dev installation

Install PyDANCE with extra dependencies for dev

```bash
pip install -e ."[dev]"
```

Make sure dependencies have specific pinned versions

```bash
pip install -r requirements.txt
```

Install pre-commit hooks for code quality checks

```bash
pre-commit install
```

### Run tests

Run all tests on current environment using pytest

```bash
pytest -v
```

Run full test from the ground up including environment set up using [tox](https://tox.wiki/en/latest/) on CPU

```bash
tox -e python3.8-cpu
```
