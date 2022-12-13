"""Reimplementation of the scDeepSort cell-type annotation method.

Reference
---------
Shao, Xin, et al. "scDeepSort: a pre-trained cell-type annotation method for single-cell transcriptomics using deep
learning with a weighted graph neural network." Nucleic acids research 49.21 (2021): e122-e122.

"""
import os
import time
from copy import deepcopy
from pathlib import Path

import dgl.function as fn
import torch
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler
from sklearn.metrics import accuracy_score

DEBUG = os.environ.get("DANCE_DEBUG")


class AdaptiveSAGE(nn.Module):
    """The AdaptiveSAGE graph convolution layer.

    Similar to SAGE convolution with mean aggregation, but with additional flexibility that adaptively assigns
    importance to gene-cell interactions, as well as gene and cell self-loops. In particular, each gene will be
    associated with an importance score `beta` that are used as aggregation weights by the cell nodes. Additionally,
    there are two `alpha` parameters indicating how much each cell or gene should retain its previous representations.

    Note
    ----
    In practice, `alpha` and `beta` are stored in a unified tensor called `alpha`. The first #gene elements of this
    tensor are the `beta` values and the last two elements are the actual `alpha` values.

    """

    def __init__(self, dim_in, dim_out, alpha, dropout_layer, act_layer, norm_layer):
        """Initialize the AdaptiveSAGE convolution layer.

        Parameters
        ----------
        dim_in : int
            Input feature dimensions.
        dim_out : int
            output feature dimensinos.
        alpha : Tensor
            Shared learnable parameters containing gene-cell interaction strengths and those for the cell and gene
            self-loops.
        dropout_layer : torch.nn
            Dropout layer.
        act_layer : torch.nn
            Activation layer.
        norm_layer : torch.nn
            Normalization layer.

        """
        super().__init__()

        self.alpha = alpha
        self.gene_num = len(alpha) - 2

        self.layers = nn.ModuleList()
        self.layers.append(dropout_layer)
        self.layers.append(nn.Linear(dim_in, dim_out))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain("relu"))
        self.layers.append(act_layer)
        self.layers.append(norm_layer)

    def message_func(self, edges):
        """Message update function.

        Reweight messages based on 1) the shared learnable interaction strengths and 2) the underlying edgeweights of
        the graph. In particular, for 1), gene-cell interaction (undirectional) will be weighted by the gene specific
        `beta` value, and the cell and gene self-interactions will be weighted based on the corresponding `alpha`
        values.

        """
        number_of_edges = edges.src["h"].shape[0]
        src_id, dst_id = edges.src["id"], edges.dst["id"]
        indices = (self.gene_num + 1) * torch.ones(number_of_edges, dtype=torch.long, device=src_id.device)
        indices = torch.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = torch.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = torch.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        if DEBUG:
            print(
                f"{((src_id >= 0) & (dst_id < 0)).sum():>10,} (geen->cell), "
                f"{((src_id < 0) & (dst_id >= 0)).sum():>10,} (cell->gene), "
                f"{((src_id >= 0) & (dst_id >= 0)).sum():>10,} (self-gene), "
                f"{((src_id < 0) & (dst_id < 0)).sum():>10,} (self-cell), ", )
        h = edges.src["h"] * self.alpha[indices]
        return {"m": h * edges.data["weight"]}

    def forward(self, block, h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata["h"] = h_src
            block.dstdata["h"] = h_dst
            block.update_all(self.message_func, fn.mean("m", "neigh"))

            z = block.dstdata["h"]
            for layer in self.layers:
                z = layer(z)

            return z


class GNN(nn.Module):
    """The scDeepSort GNN model.

    Message passing between cell and genes to learn cell representations for cell-type annotations. The graph contains
    both cell and gene nodes. The gene features are initialized as PCA embeddings from the (normalized) scRNA-seq
    samples matrix, and the cell features are computed as the weighted aggregation of the gene features according to
    each cell's gene expressions.

    """

    def __init__(self, dim_in, dim_out, dim_hid, n_layers, gene_num, activation=None, norm=None, dropout=0.):
        """Initialize the scDeepSort GNN model.

        Parameters
        ----------
        dim_in : int
            Input dimension (PCA embeddings dimension).
        dim_out : int
            Output dimension (number of unique cell labels).
        n_layers : int
            Number of convolution layers.
        gene_num : int
            Number of genes.
        dropout : float
            Dropout ratio.
        activation : torch.nn, optional
            Activation layer.
        norm : torch.nn, optional
            Normalization layer.

        """
        super().__init__()

        self.n_layers = n_layers
        self.gene_num = gene_num

        # [gene_num] is alpha of gene-gene self loop, [gene_num+1] is alpha of cell-cell self loop, the rest are betas
        self.alpha = nn.Parameter(torch.tensor([1] * (self.gene_num + 2), dtype=torch.float32).unsqueeze(-1))

        dropout_layer = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        act_layer = activation or nn.Identity()
        norm_layer = norm or nn.Identity()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            input_dimension = dim_in if i == 0 else dim_hid
            self.layers.append(AdaptiveSAGE(input_dimension, dim_hid, self.alpha, dropout_layer, act_layer, norm_layer))

        self.linear = nn.Linear(dim_hid, dim_out)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, blocks, x):
        assert len(blocks) == len(self.layers), f"Inonsistent layer info: {len(blocks)=} vs {len(self.layers)=}"
        for block, layer in zip(blocks, self.layers):
            x = layer(block, x)
        return self.linear(x)


class ScDeepSort:
    """The ScDeepSort cell-type annotation model."""

    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hid,
        num_layers,
        species,
        tissue,
        *,
        dropout=0,
        batch_size=500,
        gpu=-1,
        save_dir="result",
    ):  # yapf: disable
        self.dense_dim = dim_in
        self.hidden_dim = dim_hid
        self.n_layers = num_layers
        self.dropout = dropout
        self.species = species
        self.tissue = tissue
        self.batch_size = batch_size
        self.gpu = gpu
        self.save_dir = save_dir

        self.postfix = time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S")
        self.prj_path = Path(__file__).resolve().parents[4]
        self.save_path = (self.prj_path / "saved_models" / "single_modality" / "cell_type_annotation" / "pretrained" /
                          self.species / "models")

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device("cpu" if self.gpu == -1 else f"cuda:{gpu}")

        self.num_labels = dim_out

    def fit(self, graph, labels, epochs=300, lr=1e-3, weight_decay=0, val_ratio=0.2):
        """Train scDeepsort model.

        Parameters
        ----------
        num_cells : int
            The number of cells in the training set.
        num_genes : int
            The number of genes in the training set.
        num_labels : int
            The number of labels in the training set.
        graph : dgl.DGLGraph
            Training graph.
        train_ids : Tensor
            The training ids.
        test_ids : Tensor
            The testing ids.
        labels : Tensor
            Node (cell, gene) labels, -1 for genes.

        """
        gene_mask = graph.ndata["id"] != -1
        cell_mask = graph.ndata["id"] == -1
        num_genes = gene_mask.sum()
        num_cells = cell_mask.sum()

        perm = torch.randperm(num_cells) + num_genes
        num_val = int(num_cells * val_ratio)
        val_idx = perm[:num_val].to(self.device)
        train_idx = perm[num_val:].to(self.device)

        full_labels = -torch.ones(num_genes + num_cells, dtype=torch.long)
        full_labels[-num_cells:] = labels
        graph = graph.to(self.device)
        graph.ndata["label"] = full_labels.to(self.device)

        self.model = GNN(self.dense_dim, self.num_labels, self.hidden_dim, self.n_layers, num_genes,
                         activation=nn.ReLU(), dropout=self.dropout).to(self.device)
        print(self.model)

        self.sampler = NeighborSampler(fanouts=[-1] * self.n_layers, edge_dir="in")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

        print(f"Train Number: {len(train_idx)}, Val Number: {len(val_idx)}")
        max_val_acc, _train_acc, _epoch = 0, 0, 0
        best_state_dict = None
        for epoch in range(epochs):
            loss = self.cal_loss(graph, train_idx)
            train_acc = self.evaluate(graph, train_idx)[-1]
            val_correct, val_unsure, val_acc = self.evaluate(graph, val_idx)
            if max_val_acc <= val_acc:
                final_val_correct_num = val_correct
                final_val_unsure_num = val_unsure
                _train_acc = train_acc
                _epoch = epoch
                max_val_acc = val_acc
                self.save_model()
                best_state_dict = deepcopy(self.model.state_dict())
            print(f">>>>Epoch {epoch:04d}: Train Acc {train_acc:.4f}, Loss {loss / len(train_idx):.4f}, "
                  f"Val correct {val_correct}, Val unsure {val_unsure}, Val Acc {val_acc:.4f}")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        print(f"---{self.species} {self.tissue} Best val result:---")
        print(f"Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Val Correct Num {final_val_correct_num}, "
              f"Val Total Num {len(val_idx)}, Val Unsure Num {final_val_unsure_num}, "
              f"Val Acc {final_val_correct_num / len(val_idx):.4f}")

    def cal_loss(self, graph, idx):
        """Calculate loss.

        Returns
        -------
        float
            Loss function value.

        """
        self.model.train()
        total_loss = 0

        dataloader = DataLoader(graph=graph, indices=idx, graph_sampler=self.sampler, batch_size=self.batch_size,
                                shuffle=True)
        for _, _, blocks in dataloader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata["features"]
            output_labels = blocks[-1].dstdata["label"]
            output_predictions = self.model(blocks, input_features)

            loss = self.loss_fn(output_predictions, output_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss

    @torch.no_grad()
    def evaluate(self, graph, ids, unsure_rate: float = 2.0):
        """Evaluate the trained scDeepsort model.

        Parameters
        ----------
        ids : Tensor
            A 1-D tensor containing node IDs to be evaluated on.

        Returns
        -------
        Tuple[int, int, float]
            The total number of correct prediction, the total number of unsure prediction, and the accuracy score.

        """
        self.model.eval()
        total_correct, total_unsure = 0, 0

        dataloader = DataLoader(graph=graph, indices=ids, graph_sampler=self.sampler, batch_size=self.batch_size,
                                shuffle=True)
        for _, _, blocks in dataloader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata["features"]
            output_labels = blocks[-1].dstdata["label"]
            output_predictions = self.model(blocks, input_features)

            for pred, label in zip(output_predictions.cpu(), output_labels.cpu()):
                max_prob = pred.max().item()
                if max_prob < unsure_rate / self.num_labels:
                    total_unsure += 1
                elif pred.argmax().item() == label:
                    total_correct += 1

        return total_correct, total_unsure, total_correct / len(ids)

    def save_model(self):
        """Save the model at the save_path."""
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        torch.save(state, self.save_path / f"{self.species}-{self.tissue}.pt")

    def load_model(self):
        """Load the model from the model path."""
        filename = f"{self.species}-{self.tissue}.pt"
        model_path = self.prj_path / "pretrained" / self.species / "models" / filename
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def predict_proba(self, graph):
        """Perform inference on a test dataset.

        Parameters
        ----------
        num : int
            Test dataset number.

        Returns
        -------
        list
            Predicted labels.

        """
        self.model.eval()

        cell_mask = graph.ndata["id"] == -1
        idx = torch.where(cell_mask)[0].to(self.device)
        graph = graph.to(self.device)

        logits = torch.zeros(graph.number_of_nodes(), self.num_labels)
        dataloader = DataLoader(graph=graph, indices=idx, graph_sampler=self.sampler, batch_size=self.batch_size)
        for _, output_nodes, blocks in dataloader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata["features"]
            logits[output_nodes] = self.model(blocks, input_features).detach().cpu()

        pred_prob = nn.functional.softmax(logits[cell_mask], dim=-1).numpy()
        return pred_prob

    def predict(self, graph, unsure_rate: float = 2.0):
        """Perform prediction on all test datasets.

        Parameters
        ----------

        Returns
        -------

        """
        pred_prob = self.predict_proba(graph)

        pred = pred_prob.argmax(1)
        unsure = pred_prob.max(1) < unsure_rate / self.num_labels

        return pred, unsure

    def score(self, pred, true):
        """Compute model performance on test datasets based on accuracy.

        Parameters
        ----------
        predicted_labels : dict
            A dictionary where the keys are test dataset IDs and the values are the predicted labels.
        true_labels : dict
            A dictionary where the keys are test dataset IDs and the values are the true labels of the cells. Each
            element, i.e., the label, can be either a specific value (e.g., string or intger) or a set of values,
            allowing multiple mappings.

        Returns
        -------
        dict
            A diction of correct prediction numbers, total samples, unsured prediction numbers, and accuracy.

        """
        if true.max() == 1:
            num_samples = true.shape[0]
            return (true[range(num_samples), pred.ravel()]).sum() / num_samples
        else:
            return accuracy_score(pred, true)
