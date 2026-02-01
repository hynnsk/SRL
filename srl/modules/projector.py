from typing import Dict, List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from srl.utils import make_build_fn

@make_build_fn(__name__, "projector")
def build(config, name: str):
    pass  # No special module building needed

class Projector(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.projector = nn.Linear(inp_dim, outp_dim, bias=False)

    def forward(self, features):
        features = self.projector(features)
        return features

