import torch.nn as nn
from transformers.activations import get_activation
import copy


class ElectraClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other):
        super().__init__()
        self.dropout1 = other.dropout
        self.dense = other.dense
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.out_proj

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(
            x
        )  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x
