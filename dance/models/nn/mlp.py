import itertools

import torch
import torch.nn as nn

from dance import logger
from dance.typing import Optional, Tuple


class VanillaMLP(nn.Module):
    """Vanilla multilayer perceptron with ReLU activation.

    Parameters
    ----------
    input_dim
        Input feature dimension.
    output_dim
        Output dimension.
    hidden_dims
        Hidden layer dimensions.
    device
        Computation device.
    random_seed
        Random seed controlling the model weights initialization.

    """

    def __init__(self, input_dim: int, output_dim: int, *, hidden_dims: Tuple[int, ...] = (100, 50, 25),
                 device: str = "cpu", random_seed: Optional[int] = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.device = device
        self.random_seed = random_seed

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            *itertools.chain.from_iterable(
                zip(
                    map(nn.Linear, hidden_dims[:-1], hidden_dims[1:]),
                    itertools.repeat(nn.ReLU()),
                )),
            nn.Linear(hidden_dims[-1], output_dim),
        ).to(device)
        self.initialize_parameters()
        logger.debug(f"Initialized model:\n{self.model}")

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def initialize_parameters(self):
        """Initialize parameters."""
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        for i in range(0, len(self.model), 2):
            nn.init.xavier_normal_(self.model[i].weight)
            self.model[i].bias[:] = 0
