.. dance documentation master file, created by
   sphinx-quickstart on Mon Aug  8 12:33:49 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DANCE documentation
===================

**DANCE** is a Python toolkit to support deep learning models for analyzing
single-cell gene expression at scale. It includes three modules at present:

#. **Single-modality analysis**
#. **Single-cell multimodal omics**
#. **Spatially resolved transcriptomics**

Our goal is to build up a deep learning community for single-cell analysis and
provide GNN based architectures for users to further develop in single-cell
analysis.

Getting started
---------------

To install the DANCE package, first make sure you have correctly set up
dependencies such as
`PyTorch <https://pytorch.org/get-started/>`_,
`PyG <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_, and
`DGL <https://www.dgl.ai/pages/start.html>`_.
Then, simply install the DANCE package from
`PyPI <https://pypi.org/project/pydance/>`_ via

.. code-block:: bash

   pip install pydance

Alternatively, use our installation script provided in our
`Github <https://github.com/OmicsML/dance>`_ repository to setup a `dance`
`conda <https://conda.io/projects/conda/en/latest/index.html>`_
environment with all necessary dependencies as well as DANCE.

.. code-block:: bash

   git clone git@github.com:OmicsML/dance.git && cd dance
   source install.sh cu117  # other options are [cpu,cu118]

Finally, you can checkout some
`example scripts <https://github.com/OmicsML/dance/tree/main/examples>`_
we provided to reproduce
some of the experiments from the original papers.

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules/single_modality
   modules/multi_modality
   modules/spatial

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/data
   api/datasets
   api/transforms
   api/transforms.graph

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
