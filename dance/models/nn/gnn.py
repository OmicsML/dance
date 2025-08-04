import dgl
import torch
import torch.nn as nn

from dance import logger


class AdaptiveSAGE(nn.Module):
    """The AdaptiveSAGE graph convolution layer from scDeepSort.

    https://doi.org/10.1093/nar/gkab775

    Similar to SAGE convolution with mean aggregation, but with additional flexibility that adaptively assigns
    importance to gene-cell interactions, as well as gene and cell self-loops. In particular, each gene will be
    associated with an importance score ``beta`` that are used as aggregation weights by the cell nodes. Additionally,
    there are two ``alpha`` parameters indicating how much each cell or gene should retain its previous representations.

    Parameters
    ----------
    dim_in
        Input feature dimensions.
    dim_out
        output feature dimensions.
    alpha
        Shared learnable parameters containing gene-cell interaction strengths and those for the cell and gene
        self-loops.
    dropout_layer
        Dropout layer.
    act_layer
        Activation layer.
    norm_layer
        Normalization layer.

    Note
    ----
    In practice, ``alpha`` and ``beta`` are stored in a unified tensor called ``alpha``. The first #gene elements of
    this tensor are the ``beta`` values and the last two elements are the actual ``alpha`` values.

    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        alpha: torch.Tensor,
        dropout_layer: nn.Module,
        act_layer: nn.Module,
        norm_layer: nn.Module,
    ):
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
        ``beta`` value, and the cell and gene self-interactions will be weighted based on the corresponding ``alpha``
        values.

        """
        number_of_edges = edges.src["h"].shape[0]
        src_id, dst_id = edges.src["cell_id"], edges.dst["cell_id"]
        indices = (self.gene_num + 1) * torch.ones(number_of_edges, dtype=torch.long, device=src_id.device)
        indices = torch.where((src_id >= 0) & (dst_id < 0), src_id, indices)  # gene->cell
        indices = torch.where((dst_id >= 0) & (src_id < 0), dst_id, indices)  # cell->gene
        indices = torch.where((dst_id >= 0) & (src_id >= 0), self.gene_num, indices)  # gene-gene
        logger.debug(f"{((src_id >= 0) & (dst_id < 0)).sum():>10,} (geen->cell), "
                     f"{((src_id < 0) & (dst_id >= 0)).sum():>10,} (cell->gene), "
                     f"{((src_id >= 0) & (dst_id >= 0)).sum():>10,} (self-gene), "
                     f"{((src_id < 0) & (dst_id < 0)).sum():>10,} (self-cell), ")
        h = edges.src["h"] * self.alpha[indices]
        return {"m": h * edges.data["weight"]}

    def forward(self, block, h):
        with block.local_scope():
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata["h"] = h_src
            block.dstdata["h"] = h_dst
            block.update_all(self.message_func, dgl.function.mean("m", "neigh"))

            z = block.dstdata["h"]
            for layer in self.layers:
                z = layer(z)

            return z
