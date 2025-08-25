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

[![Slack](https://img.shields.io/badge/slack-OmicsML-brightgreen)](https://omicsml.slack.com)
[![Twitter URL](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2FOmicsML)](https://twitter.com/OmicsML)

## Overview of DANCE 1.0 and 2.0

**DANCE 1.0** is a Python toolkit designed to support deep learning models for large-scale analysis of single-cell gene expression data. Its goal is to foster a deep learning community and establish a benchmark platform for computational methods in single-cell analysis.

**DANCE 2.0** extends this effort by introducing an automated preprocessing recommendation platform. It addresses the pressing need to move beyond trial-and-error approaches by transforming single-cell preprocessing into a systematic, data-driven, and interpretable workflow.

Both include three modules at present:

1. **Single-modality analysis**: cell type annotation, clustering, gene imputation
1. **Single-cell multimodal omics**: modality prediction (only DANCE 1.0), modality matching(only DANCE 1.0), joint embedding
1. **Spatially resolved transcriptomics**: spatial domain identification, cell type deconvolution

## DANCE 2.0 Release Schedule

- [ ] Open-source release of the DANCE 2.0 codebase
- [ ] Launch of the DANCE 2.0 web platform for users to upload datasets and receive optimal preprocessing recommendations
- [ ] Release of the DANCE 2.0 API for programmatic access to preprocessing recommendations

## Useful links

**DANCE Open Source**: https://github.com/OmicsML/dance \
**DANCE Documentation**: https://pydance.readthedocs.io/en/latest/ \
**DANCE 1.0 Tutorial**: https://github.com/OmicsML/dance-tutorials \
**DANCE 1.0 Paper (published on Genome Biology)**: [DANCE: a deep learning library and benchmark platform for single-cell analysis](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03211-z) \
**DANCE 2.0 Paper**: [DANCE 2.0: Transforming single-cell analysis from black box to transparent workflow](https://www.biorxiv.org/content/10.1101/2025.07.17.665427v1) \
**Survey Paper (published on ACM TIST):** [Deep Learning in Single-cell Analysis](https://dl.acm.org/doi/10.1145/3641284)

## Join the Community

Slack: https://join.slack.com/t/omicsml/shared_invite/zt-1hxdz7op3-E5K~EwWF1xDvhGZFrB9AbA \
Twitter: https://twitter.com/OmicsML \
Wechat Group Assistant: 736180290 \
Email: danceteamgnn@gmail.com

## Contributing

Community-wide contribution is the key to sustainable development and
continual growth of the DANCE package. We deeply appreciate any contribution
made to improve the DANCE code base. If you would like to get started, please
refer to our brief [guidelines](CONTRIBUTING.md) about our automated quality
controls, as well as setting up the `dev` environments.

## Citation

If you find our work useful in your research, please consider citing our DANCE package or survey paper:

```bibtex
@article{ding2025dance,
  title={DANCE 2.0: Transforming single-cell analysis from black box to transparent workflow},
  author={Ding, Jiayuan and Xing, Zhongyu and Wang, Yixin and Liu, Renming and Liu, Sheng and Huang, Zhi and Tang, Wenzhuo and Xie, Yuying and Zou, James and Qiu, Xiaojie and others},
  journal={bioRxiv},
  pages={2025--07},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

```bibtex
@article{ding2024dance,
  title={DANCE: A deep learning library and benchmark platform for single-cell analysis},
  author={Ding, Jiayuan and Liu, Renming and Wen, Hongzhi and Tang, Wenzhuo and Li, Zhaoheng and Venegas, Julian and Su, Runze and Molho, Dylan and Jin, Wei and Wang, Yixin and others},
  journal={Genome Biology},
  volume={25},
  number={1},
  pages={1--28},
  year={2024},
  publisher={BioMed Central}
}
```

```bibtex
@article{molho2024deep,
  title={Deep learning in single-cell analysis},
  author={Molho, Dylan and Ding, Jiayuan and Tang, Wenzhuo and Li, Zhaoheng and Wen, Hongzhi and Wang, Yixin and Venegas, Julian and Jin, Wei and Liu, Renming and Su, Runze and others},
  journal={ACM Transactions on Intelligent Systems and Technology},
  volume={15},
  number={3},
  pages={1--62},
  year={2024},
  publisher={ACM New York, NY}
}
```

## Usage (DANCE 1.0)

### Overview

In release 1.0, the main usage of the DANCE is to provide readily available experiment reproduction
(see detail information about the reproduced performance [below](#implemented-algorithms)).
Users can easily reproduce selected experiments presented in the original papers for the computational single-cell methods implemented in DANCE, which can be found under [`examples/`](examples).

### Motivation

Computational methods for single-cell analysis are quickly emerging, and the field is revolutionizing the usage of single-cell data to gain biological insights.
A key challenge to continually developing computational single-cell methods that achieve new state-of-the-art performance is reproducing previous benchmarks.
More specifically, different studies prepare their datasets and perform evaluation differently,
and not to mention the compatibility of different methods, as they could be written in different languages or using incompatible library versions.

DANCE addresses these challenges by providing a unified Python package implementing many popular computational single-cell methods (see [Implemented Algorithms](#implemented-algorithms)),
as well as easily reproducible experiments by providing unified tools for

- Data downloading
- Data (pre-)processing and transformation (e.g. graph construction)
- Model training and evaluation

### Example: run cell-type annotation benchmark using scDeepSort

- Step0. Install DANCE (see [Installation](#installation))
- Step1. Navigate to the folder containing the corresponding example scrtip.
  In this case, it is [`examples/single_modality/cell_type_annotation`](examples/single_modality/cell_type_annotation).
- Step2. Obtain command line interface (CLI) options for a particular experiment to reproduce at the end of the
  [script](examples/single_modality/cell_type_annotation/scdeepsort.py).
  For example, the CLI options for reproducing the `Mouse Brain` experiment is
  ```bash
  python scdeepsort.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695
  ```
- Step3. Wait for the experiment to finish and check results.

## Usage (DANCE 2.0)

### Overview

In release 2.0, DANCE evolves from an experiment reproduction library into an automated and interpretable preprocessing platform. It provides powerful tools to optimize your single-cell analysis workflows:
To discover the best preprocessing pipeline for a specific method, you can use our Method-Aware Preprocessing (MAP) module. For practical examples on how to run this locally, please see [`examples/tuning/custom-methods/`](examples/tuning/custom-methods%60).
To get an instant, high-quality pipeline recommendation for a new dataset, you can use our Dataset-Aware Preprocessing (DAP) web service, available at http://omicsml.ai:81/dance/.
Together, these features transform single-cell preprocessing from a manual, trial-and-error process into a systematic, data-driven, and reproducible workflow.

### Motivation

While DANCE 1.0 addressed benchmark reproduction, a more fundamental challenge in single-cell analysis is the preprocessing itself. The optimal combination of normalization, gene selection, and dimensionality reduction varies across tasks, models, and datasets, yet the selection process is often guided by legacy defaults or time-consuming trial-and-error. This inconsistency hinders reproducibility and can lead to suboptimal or even misleading results.
DANCE 2.0 tackles this challenge by transforming preprocessing from a black-box art into a systematic, data-driven science. It provides tools to automatically construct pipelines tailored to a specific analytical method and dataset, ensuring more robust and transparent downstream analysis.

### Example: run cell-type annotation benchmark using SVM

- Step0. Install DANCE (see [Installation](#installation))
- Step1. Navigate to the folder containing the corresponding example scrtip.
  In this case, it is [`examples/tuning/cta_svm`](examples/tuning/cta_svm).
- Step2. Obtain command line interface (CLI) options for a particular experiment to reproduce at the end of the
  [script](examples/tuning/cta_svm/main.py).
  For example, the CLI options for reproducing the `Human Brain` experiment is
  ```bash
  python main.py --tune_mode (pipeline/params/pipeline_params) --species human --tissue Brain --train_dataset 328 --test_dataset 138 --valid_dataset 328
  ```
- Step3. Wait for the experiment to finish and check results.

## Installation

<H3>Quick install</H3>

The full installation process might be a bit tedious and could involve some debugging when using CUDA enabled packages.
Thus, we provide an `install.sh` script that simplifies the installation process, assuming the user have [conda](https://conda.io/projects/conda/en/latest/index.html) set up on their machines.
The installation script creates a conda environment `dance` and install the DANCE package along with all its dependencies with a apseicifc CUDA version.
Currently, two options are accepted: `cpu` and `cu118`.
For example, to install the DANCE package using CUDA 11.8 in a `dance-env` conda environment, simply run:

```bash
# Clone the repository via SSH
git clone git@github.com:OmicsML/dance.git && cd dance
# Alternatively, use HTTPS if you have not set up SSH
# git clone https://github.com/OmicsML/dance.git  && cd dance

# Run the auto installation script to install DANCE and its dependencies in a conda environment
source install.sh cu118 dance-env
```

**Note**: the first argument for cuda version is mandatory, while the second argument for conda environment name is optional (default is `dance`).

<H3>Custom install</H3>
<br>

**Step1. Setup environment**

First create a conda environment for dance (optional)

```bash
conda create -n dance python=3.11 -y && conda activate dance
```

Then, install CUDA enabled packages (PyTorch, PyG, DGL):

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.4.0
pip install dgl==1.1.3 -f https://data/dgl.ai/wheels/cu118/repo.html
```

Alternatively, install these dependencies for CPU only:

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.4.0
pip install dgl==1.1.3 -f https://data/dgl.ai/wheels/repo.html
```

For more information about installation or other CUDA version options, check out the installation pages for the corresponding packages

- [PyTorch](https://pytorch.org/get-started/)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [DGL](https://www.dgl.ai/pages/start.html)

**Step2. Install DANCE**

Install from PyPI

```bash
pip install pydance
```

Or, install the latest dev version from source

```bash
git clone https://github.com/OmicsML/dance.git && cd dance
pip install -e .
```

</details>

## Implemented Algorithms

**P1** not covered in the first release

### Single Modality Module

#### 1）Imputation

| BackBone            | Model        | Algorithm                                                                                                    | Year | CheckIn |
| ------------------- | ------------ | ------------------------------------------------------------------------------------------------------------ | ---- | ------- |
| GNN                 | GraphSCI     | Imputing Single-cell RNA-seq data by combining Graph Convolution and Autoencoder Neural Networks             | 2021 | ✅      |
| GNN                 | scGNN (2020) | SCGNN: scRNA-seq Dropout Imputation via Induced Hierarchical Cell Similarity Graph                           | 2020 | P1      |
| GNN                 | scGNN (2021) | scGNN is a novel graph neural network framework for single-cell RNA-Seq analyses                             | 2021 | ✅      |
| GNN                 | GNNImpute    | An efficient scRNA-seq dropout imputation method using graph attention network                               | 2021 | P1      |
| Graph Diffusion     | MAGIC        | MAGIC: A diffusion-based imputation method reveals gene-gene interactions in single-cell RNA-sequencing data | 2018 | P1      |
| Probabilistic Model | scImpute     | An accurate and robust imputation method scImpute for single-cell RNA-seq data                               | 2018 | P1      |
| GAN                 | scGAIN       | scGAIN: Single Cell RNA-seq Data Imputation using Generative Adversarial Networks                            | 2019 | P1      |
| NN                  | DeepImpute   | DeepImpute: an accurate, fast, and scalable deep neural network method to impute single-cell RNA-seq data    | 2019 | ✅      |
| NN + TF             | Saver-X      | Transfer learning in single-cell transcriptomics improves data denoising and pattern discovery               | 2019 | P1      |

| Model      | Mouse Brain (DANCE 2.0/DANCE1.0/Original) | Mouse Embryo (DANCE 2.0/DANCE1.0/Original) | PBMC (DANCE 2.0/DANCE1.0/Original) | Evaluation Metric |
| ---------- | ----------------------------------------- | ------------------------------------------ | ---------------------------------- | ----------------- |
| DeepImpute | 0.229/0.244/NA                            | 0.252/0.255/NA                             | 0.220/0.230/NA                     | Test MRE          |
| GraphSCI   | 0.453/0.654/NA                            | 0.459/0.497/NA                             | 0.458/0.704/NA                     | Test MRE          |
| scGNN2     | 0.323/0.629/NA                            | 0.299/0.620/NA                             | 0.441/0.684/NA                     | Test MRE          |

**Note**: Stage 1, 2 and 3 (valid mask as metric for selection) for all methods.

**Note**: scGNN2.0 is evaluated on 2,000 genes with highest variance following the original paper.

#### 2）Cell Type Annotation

| BackBone                | Model         | Algorithm                                                                                                        | Year | CheckIn |
| ----------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN                     | ScDeepsort    | Single-cell transcriptomics with weighted GNN                                                                    | 2021 | ✅      |
| Logistic Regression     | Celltypist    | Cross-tissue immune cell analysis reveals tissue-specific features in humans.                                    | 2021 | ✅      |
| Random Forest           | singleCellNet | SingleCellNet: a computational tool to classify single cell RNA-Seq data across platforms and across species.    | 2019 | ✅      |
| Neural Network          | ACTINN        | ACTINN: automated identification of cell types in single cell RNA sequencing.                                    | 2020 | ✅      |
| Hierarchical Clustering | SingleR       | Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage.           | 2019 | P1      |
| SVM                     | SVM           | A comparison of automatic cell identification methods for single-cell RNA sequencing data.                       | 2018 | ✅      |
| GNN                     | scHeteroNet   | scHeteroNet: A Heterophily-Aware Graph Neural Network for Accurate Cell Type Annotation and Novel Cell Detection | 2025 | ✅      |

| Model         | GSE67835 Brain<br>(DANCE 2.0/DANCE 1.0/Original) | CD8+ TIL atlas<br>(DANCE 2.0/DANCE 1.0/Original) | GSE123813 Immune<br>(DANCE 2.0/DANCE 1.0/Original) | CD4+ TIL atlas<br>(DANCE 2.0/DANCE 1.0/Original) | GSE182320 (Tissue- Spleen)<br>(DANCE 2.0/DANCE 1.0/Original) | Evaluation Metric |
| ------------- | ------------------------------------------------ | ------------------------------------------------ | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| SVM           | 0.82/0.07/NA                                     | 0.81/0.39/NA                                     | 0.86/0.83/NA                                       | 0.92/0.48/NA                                     | 0.47/0.30/NA                                                 | ACC               |
| ACTINN        | 0.80/0.80/NA                                     | 0.84/0.78/NA                                     | 0.83/0.81/NA                                       | 0.92/0.89/NA                                     | 0.47/0.44/NA                                                 | ACC               |
| singleCellNet | 0.78/0.77/NA                                     | 0.76/0.75/NA                                     | 0.85/0.84/NA                                       | 0.87/0.85/NA                                     | 0.45/0.44/NA                                                 | ACC               |
| Celltypist    | 0.84/0.90/NA                                     | 0.81/0.72/NA                                     | 0.83/0.80/NA                                       | 0.92/0.87/NA                                     | 0.45/0.43/NA                                                 | ACC               |
| ScdeepSort    | 0.84/0.07/NA                                     | 0.83/0.65/NA                                     | 0.83/0.82/NA                                       | 0.92/0.78/NA                                     | 0.45/0.43/NA                                                 | ACC               |
| scHeteroNet   | 0.87/0.83/NA                                     | 0.80/0.78/NA                                     | 0.82/0.81/NA                                       | 0.91/0.89/NA                                     | 0.47/0.45/NA                                                 | ACC               |

**Note**: Stage 1, 2 and 3 (valid dataset as metric for selection) for all methods.

#### 3）Clustering

| BackBone    | Model         | Algorithm                                                                                                    | Year | CheckIn |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------ | ---- | ------- |
| GNN         | graph-sc      | GNN-based embedding for clustering scRNA-seq data                                                            | 2022 | ✅      |
| GNN         | scTAG         | ZINB-based Graph Embedding Autoencoder for Single-cell RNA-seq Interpretations                               | 2022 | ✅      |
| GNN         | scDSC         | Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph neural network | 2022 | ✅      |
| GNN         | scGAC         | scGAC: a graph attentional architecture for clustering single-cell RNA-seq data                              | 2022 | P1      |
| AutoEncoder | scDeepCluster | Clustering single-cell RNA-seq data with a model-based deep learning approach                                | 2019 | ✅      |
| AutoEncoder | scDCC         | Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data                   | 2021 | ✅      |
| AutoEncoder | scziDesk      | Deep soft K-means clustering with self-training for single-cell RNA sequence data                            | 2020 | P1      |

| Model         | Worm Neuron (DANCE 2.0/DANCE 1.0/Original) | Mouse Bladder (DANCE 2.0/DANCE 1.0/Original) | 10X PBMC (DANCE 2.0/DANCE 1.0/Original) | Mouse ES (DANCE 2.0/DANCE 1.0/Original) | Evaluation Metric |
| ------------- | ------------------------------------------ | -------------------------------------------- | --------------------------------------- | --------------------------------------- | ----------------- |
| graph-sc      | 0.71/0.53/0.46                             | 0.76/0.59/0.63                               | 0.79/0.68/0.70                          | 0.95/0.81/0.78                          | ARI               |
| scDCC         | 0.69//0.41/0.58                            | 0.78/0.60/0.66                               | 0.84/0.82/0.81                          | 0.9987/0.98/NA                          | ARI               |
| scDeepCluster | 0.70/0.51/0.52                             | 0.80/0.56/0.58                               | 0.83/0.81/0.78                          | 0.9951/0.98/0.97                        | ARI               |
| scDSC         | 0.66/0.46/0.65                             | 0.68/0.65/0.72                               | 0.72/0.72/0.78                          | 0.98/0.98/0.84/NA                       | ARI               |
| scTAG         | 0.72/0.49/NA                               | 0.76/0.69/NA                                 | 0.81/0.77/NA                            | 0.93/0.96/NA                            | ARI               |

**Note**: Stage 1, 2 and 3 (test dataset as metric for selection) for all methods.

### Multimodality Module

#### 1）Modality Prediction

| BackBone         | Model                    | Algorithm                                                                                          | Year | CheckIn |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN                  | Graph Neural Networks for Multimodal Single-Cell Data Integration                                  | 2022 | ✅      |
| GNN              | ScMoLP                   | Link Prediction Variant of ScMoGCN                                                                 | 2022 | P1      |
| GNN              | GRAPE                    | Handling Missing Data with Graph Representation Learning                                           | 2020 | P1      |
| Generative Model | SCMM                     | SCMM: MIXTURE-OF-EXPERTS MULTIMODAL DEEP GENERATIVE MODEL FOR SINGLE-CELL MULTIOMICS DATA ANALYSIS | 2021 | ✅      |
| Auto-encoder     | Cross-modal autoencoders | Multi-domain translation between single-cell imaging and sequencing data using autoencoders        | 2021 | ✅      |
| Auto-encoder     | BABEL                    | BABEL enables cross-modality translation between multiomic profiles at single-cell resolution      | 2021 | ✅      |

| Model                    | Evaluation Metric | GEX2ADT (DANCE 1.0/Original) | ADT2GEX (DANCE 1.0/Original) | GEX2ATAC (DANCE 1.0/Original) | ATAC2GEX (DANCE 1.0/Original) |
| ------------------------ | ----------------- | ---------------------------- | ---------------------------- | ----------------------------- | ----------------------------- |
| ScMoGCN                  | RMSE              | 0.3885 / 0.3885              | 0.3242 / 0.3242              | 0.1778 / 0.1778               | 0.2315 / 0.2315               |
| SCMM                     | RMSE              | 0.6264 / N/A                 | 0.4458 / N/A                 | 0.2163 / N/A                  | 0.3730 / N/A                  |
| Cross-modal autoencoders | RMSE              | 0.5725 / N/A                 | 0.3585 / N/A                 | 0.1917 / N/A                  | 0.2551 / N/A                  |
| BABEL                    | RMSE              | 0.4335 / N/A                 | 0.3673 / N/A                 | 0.1816 / N/A                  | 0.2394 / N/A                  |

#### 2) Modality Matching

| BackBone         | Model                    | Algorithm                                                                                          | Year | CheckIn |
| ---------------- | ------------------------ | -------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN                  | Graph Neural Networks for Multimodal Single-Cell Data Integration                                  | 2022 | ✅      |
| GNN/Auto-ecnoder | GLUE                     | Multi-omics single-cell data integration and regulatory inference with graph-linked embedding      | 2021 | P1      |
| Generative Model | SCMM                     | SCMM: MIXTURE-OF-EXPERTS MULTIMODAL DEEP GENERATIVE MODEL FOR SINGLE-CELL MULTIOMICS DATA ANALYSIS | 2021 | ✅      |
| Auto-encoder     | Cross-modal autoencoders | Multi-domain translation between single-cell imaging and sequencing data using autoencoders        | 2021 | ✅      |

| Model                    | Evaluation Metric | GEX2ADT (DANCE 1.0/Original) | GEX2ATAC (DANCE 1.0/Original) |
| ------------------------ | ----------------- | ---------------------------- | ----------------------------- |
| ScMoGCN                  | Accuracy          | 0.0827 / 0.0810              | 0.0600 / 0.0630               |
| SCMM                     | Accuracy          | 0.005 / N/A                  | 5e-5 / N/A                    |
| Cross-modal autoencoders | Accuracy          | 0.0002 / N/A                 | 0.0002 / N/A                  |

#### 3) Joint Embedding

| BackBone         | Model   | Algorithm                                                                                             | Year | CheckIn |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN              | ScMoGCN | Graph Neural Networks for Multimodal Single-Cell Data Integration                                     | 2022 | ✅      |
| Auto-encoder     | scMVAE  | Deep-joint-learning analysis model of single cell transcriptome and open chromatin accessibility data | 2020 | ✅      |
| Auto-encoder     | scDEC   | Simultaneous deep generative modelling and clustering of single-cell genomic data                     | 2021 | ✅      |
| GNN/Auto-ecnoder | GLUE    | Multi-omics single-cell data integration and regulatory inference with graph-linked embedding         | 2021 | P1      |
| Auto-encoder     | DCCA    | Deep cross-omics cycle attention model for joint analysis of single-cell multi-omics data             | 2021 | ✅      |

| Model   | BRAIN ATAC2GEX (DANCE 2.0/DANCE 1.0/Original) | SKIN ATAC2GEX (DANCE 2.0/DANCE 1.0/Original) | OP 2022 Multi ATAC2GEX (DANCE 2.0/DANCE 1.0/Original) | Evaluation Metric |
| ------- | --------------------------------------------- | -------------------------------------------- | ----------------------------------------------------- | ----------------- |
| DCCA    | 0.399/0.112/NA                                | 0.597/0.335/NA                               | 0.549/0.438/NA                                        | ARI               |
| scDEC   | 0.853/0.475/NA                                | 0.889/0.34/NA                                | 0.827/0.428/NA                                        | ARI               |
| ScMoGCN | 0.704/0.478/NA                                | 0.634/0.32/NA                                | 0.85/0.433/NA                                         | ARI               |
| scMVAE  | 0.342/0.218/NA                                | 0.399/0.341/NA                               | 0.437/0.362/NA                                        | ARI               |

**Note**: Stage 1, 2 and 3 (test dataset as metric for selection) for all methods.

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
| GNN                              | SpaGCN     | SpaGCN: Integrating gene expression, spatial location and histology to identify spatial domains and spatially variable genes by graph convolutional network                   | 2021 | ✅      |
| GNN                              | STAGATE    | Deciphering spatial domains from spatially resolved transcriptomics with adaptive graph attention auto-encoder                                                                | 2021 | ✅      |
| Bayesian                         | BayesSpace | Spatial transcriptomics at subspot resolution with BayesSpace                                                                                                                 | 2021 | P1      |
| Pseudo-space-time (PST) Distance | stLearn    | stLearn: integrating spatial location, tissue morphology and gene expression to find cell types, cell-cell interactions and spatial trajectories within undissociated tissues | 2020 | ✅      |
| Heuristic                        | Louvain    | Fast unfolding of community hierarchies in large networks                                                                                                                     | 2008 | ✅      |
| GNN                              | EfNST      | A composite scaling network of EfficientNet for improving spatial domain identification performance                                                                           | 2024 | ✅      |

| Model   | 151676 (DANCE 2.0/DANCE 1.0/Original) | Sub MBA (DANCE 2.0/DANCE 1.0/Original) | Evaluation Metric |
| ------- | ------------------------------------- | -------------------------------------- | ----------------- |
| Louvain | 0.27/0.25/NA                          | 0.43/0.42/NA                           | ARI               |
| SpaGCN  | 0.47/0.27/0.35                        | 0.32/0.32/NA                           | ARI               |
| STAGATE | 0.60/0.60/0.55                        | 0.29/0.27/NA                           | ARI               |
| stLearn | 0.30/0.29/NA                          | 0.45/0.36/NA                           | ARI               |
| EfNST   | 0.52/0.33/0.51                        | 0.30/0.19/NA                           | ARI               |

**Note**: Stage 1, 2 and 3 (test dataset as metric for selection) for all methods.

#### 2）Cell Type Deconvolution

| BackBone                   | Model        | Algorithm                                                                                                     | Year | CheckIn |
| -------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| GNN                        | DSTG         | DSTG: deconvoluting spatial transcriptomics data through graph-based artificial intelligence                  | 2021 | ✅      |
| logNormReg                 | SpatialDecon | Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data       | 2022 | ✅      |
| NNMFreg                    | SPOTlight    | SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics spots with single-cell transcriptomes | 2021 | ✅      |
| NN Linear + CAR assumption | CARD         | Spatially informed cell-type deconvolution for spatial transcriptomics                                        | 2022 | ✅      |
| GNN                        | STdGCN       | STdGCN: spatial transcriptomic cell-type deconvolution using graph convolutional networks                     | 2024 | ✅      |

| Model        | CARD Synthetic (DANCE 2.0/DANCE 1.0/Original) | SPOTlight Synthetic (DANCE 2.0/DANCE 1.0/Original) | Evaluation Metric |
| ------------ | --------------------------------------------- | -------------------------------------------------- | ----------------- |
| CARD         | 0.00553/0.00627/NA                            | 0.00653/0.00772/NA                                 | Test MSE          |
| DSTG         | 0.0105/0.0239/NA                              | 0.0314/0.0315/NA                                   | Test MSE          |
| SpatialDecon | 0.00754/0.00821/NA                            | 0.00528/0.00528/NA                                 | Test MSE          |
| SPOTlight    | 0.0113/0.0250/NA                              | 0.00614/0.0106/NA                                  | Test MSE          |
| STdGCN       | 0.0058/0.0202/NA                              | 0.0145/0.0261/NA                                   | Test MSE          |

**Note**: DANCE 2.0 indicates Stage 1, 2, and 3 (valid pseudo or dataset as metric for selection) for all methods.

### A Note on Function Naming

The function `ScaleFeature` has been renamed to `ColumnSumNormalize` in the code to resolve a naming ambiguity. However, historical WandB logs have not been modified and will still reference the old name (`ScaleFeature`). This is a naming change only and does not affect the program's execution.
