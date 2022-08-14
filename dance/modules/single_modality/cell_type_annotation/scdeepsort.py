"""Reimplementation of the scDeepSort cell-type annotation method.

Reference
---------
Shao, Xin, et al. "scDeepSort: a pre-trained cell-type annotation method for single-cell transcriptomics using deep
learning with a weighted graph neural network." Nucleic acids research 49.21 (2021): e122-e122.

"""
import os
import time
from pathlib import Path

import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.dataloading import DataLoader, NeighborSampler

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
        indices = np.expand_dims(np.array([self.gene_num + 1] * number_of_edges, dtype=np.int32), axis=1)
        src_id, dst_id = edges.src["id"].cpu().numpy(), edges.dst["id"].cpu().numpy()
        indices = np.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = np.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = np.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        if DEBUG:
            print(
                f"{((src_id >= 0) & (dst_id < 0)).sum():>10,} (geen->cell), "
                f"{((src_id < 0) & (dst_id >= 0)).sum():>10,} (cell->gene), "
                f"{((src_id >= 0) & (dst_id >= 0)).sum():>10,} (self-gene), "
                f"{((src_id < 0) & (dst_id < 0)).sum():>10,} (self-cell), ", )
        h = edges.src["h"] * self.alpha[indices.squeeze()]
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
    """The ScDeepSort cell-type annotation model.

    Parameters
    ----------
    params : argparse.Namespace
        A Namespace contains arguments of Scdeepsort. See parser documnetation for more details.

    """

    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S")
        self.prj_path = Path(__file__).resolve().parents[4]
        # TODO: change the prefix from `example` to `saved_models`
        self.save_path = (self.prj_path / "example" / "single_modality" / "cell_type_annotation" / "pretrained" /
                          self.params.species / "models")

        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device("cpu" if self.params.gpu == -1 else f"cuda:{params.gpu}")

        # Define the variables during training
        self.num_cells = None
        self.num_genes = None
        self.num_labels = None
        self.graph = None
        self.train_ids = None
        self.test_ids = None
        self.labels = None

        # Define the variables during prediction
        self.id2label = None
        self.test_dict = None

    def fit(self, num_cells, num_genes, num_labels, graph, train_ids, test_ids, labels):
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
        self.num_cells = num_cells
        self.num_genes = num_genes
        self.num_labels = num_labels

        self.train_ids = train_ids.to(self.device)
        self.test_ids = test_ids.to(self.device)
        self.labels = labels.to(self.device)
        self.graph = graph.to(self.device)
        self.graph.ndata["label"] = self.labels

        self.model = GNN(self.params.dense_dim, self.num_labels, self.params.hidden_dim, self.params.n_layers,
                         self.num_genes, activation=nn.ReLU(), dropout=self.params.dropout).to(self.device)
        print(self.model)

        self.sampler = NeighborSampler(fanouts=[-1] * self.params.n_layers, edge_dir="in")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr,
                                          weight_decay=self.params.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        if self.params.num_neighbors == 0:
            self.num_neighbors = self.num_cells + self.num_genes
        else:
            self.num_neighbors = self.params.num_neighbors

        print(f"Train Number: {len(self.train_ids)}, Test Number: {len(self.test_ids)}")
        max_test_acc, _train_acc, _epoch = 0, 0, 0
        for epoch in range(self.params.n_epochs):
            loss = self.cal_loss()
            train_acc = self.evaluate(self.train_ids)[-1]
            test_correct, test_unsure, test_acc = self.evaluate(self.test_ids)
            if max_test_acc <= test_acc:
                final_test_correct_num = test_correct
                final_test_unsure_num = test_unsure
                _train_acc = train_acc
                _epoch = epoch
                max_test_acc = test_acc
                self.save_model()
            print(f">>>>Epoch {epoch:04d}: Train Acc {train_acc:.4f}, Loss {loss / len(self.train_ids):.4f}, "
                  f"Test correct {test_correct}, Test unsure {test_unsure}, Test Acc {test_acc:.4f}")

        print(f"---{self.params.species} {self.params.tissue} Best test result:---")
        print(f"Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Test Correct Num {final_test_correct_num}, "
              f"Test Total Num {len(self.test_ids)}, Test Unsure Num {final_test_unsure_num}, "
              f"Test Acc {final_test_correct_num / len(self.test_ids):.4f}")

    def cal_loss(self):
        """Calculate loss.

        Returns
        -------
        float
            Loss function value.

        """
        self.model.train()
        total_loss = 0

        dataloader = DataLoader(graph=self.graph, indices=self.train_ids, graph_sampler=self.sampler,
                                batch_size=self.params.batch_size, shuffle=True)
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
    def evaluate(self, ids):
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

        dataloader = DataLoader(graph=self.graph, indices=ids, graph_sampler=self.sampler,
                                batch_size=self.params.batch_size, shuffle=True)
        for _, _, blocks in dataloader:
            blocks = [b.to(self.device) for b in blocks]
            input_features = blocks[0].srcdata["features"]
            output_labels = blocks[-1].dstdata["label"]
            output_predictions = self.model(blocks, input_features)

            for pred, label in zip(output_predictions.cpu(), output_labels.cpu()):
                max_prob = pred.max().item()
                if max_prob < self.params.unsure_rate / self.num_labels:
                    total_unsure += 1
                elif pred.argmax().item() == label:
                    total_correct += 1

        return total_correct, total_unsure, total_correct / len(ids)

    def save_model(self):
        """Save the model at the save_path."""
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
        torch.save(state, self.save_path / f"{self.params.species}-{self.params.tissue}.pt")

    def load_model(self):
        """Load the model from the model path."""
        filename = f"{self.params.species}-{self.params.tissue}.pt"
        model_path = self.prj_path / "pretrained" / self.params.species / "models" / filename
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def inference(self, num):
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

        unsure_threshold = self.params.unsure_rate / self.num_labels

        graph = self.test_dict["graph"][num].to(self.device)
        test_indices = self.test_dict["nid"][num].to(self.device)
        new_logits = torch.zeros((graph.number_of_nodes(), self.num_labels))

        dataloader = DataLoader(graph=graph, indices=test_indices, graph_sampler=self.sampler,
                                batch_size=self.params.batch_size, shuffle=True)
        for _, output_nodes, blocks in dataloader:
            blocks = [b.to(torch.device(self.device)) for b in blocks]
            input_features = blocks[0].srcdata["features"]
            new_logits[output_nodes] = self.model(blocks, input_features).detach().cpu()

        new_logits = new_logits[self.test_dict["mask"][num]]
        new_logits = nn.functional.softmax(new_logits, dim=1).numpy()
        predict_label = []
        for pred in new_logits:
            pred_label = self.id2label[pred.argmax().item()]
            predict_label.append("unsure" if pred.max().item() < unsure_threshold else pred_label)
        return predict_label

    def predict(self, id2label, test_dict):
        """Perform prediction on all test datasets.

        Parameters
        ----------
        id2label : np.ndarray
            Id to label diction.
        test_dict : dict
            The test dictionary.

        Returns
        -------
        dict
            A dictionary where the keys are the test dataset IDs and the values are the corresponding predictions.

        """
        self.id2label = id2label
        self.test_dict = test_dict
        return {num: self.inference(num) for num in self.params.test_dataset}

    @torch.no_grad()
    def score(self, predicted_labels, true_labels):
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
        output_score = {}
        for num in set(predicted_labels) & set(true_labels):
            total_num = len(predicted_labels[num])
            unsure_num = correct = 0
            for pred, true in zip(predicted_labels[num], true_labels[num]):
                if pred == "unsure":
                    unsure_num += 1
                elif pred == true or pred in true:  # either a single mapping or multiple mappings
                    correct += 1

            output_score[num] = {
                "Total number of predictions": total_num,
                "Number of correct predictions": correct,
                "Number of unsure predictions": unsure_num,
                "Accuracy": correct / total_num,
            }

        return output_score

    def save_pred(self, num, pred):
        """Save predictions for a particular test dataset.

        Parameters
        ----------
        num : int
            Test file number.
        pred : list np.array or dataframe
            Predicted labels.

        """
        label_map = pd.read_excel("./map/celltype2subtype.xlsx", sheet_name=self.params.species, header=0,
                                  names=["species", "old_type", "new_type", "new_subtype"])
        label_map = label_map.fillna("N/A", inplace=False)
        oldtype2newtype = {}
        oldtype2newsubtype = {}
        for _, old_type, new_type, new_subtype in label_map.itertuples(index=False):
            oldtype2newtype[old_type] = new_type
            oldtype2newsubtype[old_type] = new_subtype

        save_path = self.prj_path / self.params.save_dir
        if not save_path.exists():
            save_path.mkdir()
        if self.params.score:
            df = pd.DataFrame({
                "index": self.test_dict["origin_id"][num],
                "original label": self.test_dict["label"][num],
                "cell_type": [oldtype2newtype.get(p, p) for p in pred],
                "cell_subtype": [oldtype2newsubtype.get(p, p) for p in pred]
            })
        else:
            df = pd.DataFrame({
                "index": self.test_dict["origin_id"][num],
                "cell_type": [oldtype2newtype.get(p, p) for p in pred],
                "cell_subtype": [oldtype2newsubtype.get(p, p) for p in pred]
            })
        df.to_csv(save_path / (self.params.species + f"_{self.params.tissue}_{num}.csv"), index=False)
        print(f"output has been stored in {self.params.species}_{self.params.tissue}_{num}.csv")
