import json
import logging
import os
from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn

logging.basicConfig()
logger = logging.getLogger("adapters")


class BaseAdapter(nn.Module):

    @abstractmethod
    def get_config_dict(self) -> Dict:
        """Get config dict."""

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass."""

    def save(self, save_dir: str, **kwargs) -> None:
        """Save model."""
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_adapter_model.bin"))
        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, load_path: str) -> "BaseAdapter":
        """Load model."""
        with open(os.path.join(load_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = cls(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(load_path, "pytorch_adapter_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        logger.info(f"Model loaded from {load_path}")
        return model


# Define Linear Adapter Module
class LinearAdapter(BaseAdapter):
    def __init__(self, embedding_dim: int, adapter_dim: int, bias: bool) -> None:
        super(LinearAdapter, self).__init__()
        self.embedding_dim = embedding_dim
        self.adapter_dim = adapter_dim
        self.bias = bias

        # Linear adapter layers
        self.linear = nn.Linear(embedding_dim, adapter_dim, bias=bias)
        self.linear.weight.data.copy_(torch.eye(embedding_dim, adapter_dim))
        if bias:
            self.linear.bias.data.copy_(torch.zeros(adapter_dim))

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass (Wv)."""
        embed = features["token_embeddings"]
        features['token_embeddings'] = self.linear(embed)
        return features

    def get_config_dict(self) -> Dict:
        return {
            "embedding_dim": self.embedding_dim,
            "adapter_dim": self.adapter_dim,
            "bias": self.bias
        }


# Define 2-layer Adapter Module
class TwoLayerAdapter(BaseAdapter):
    def __init__(self,
                 embedding_dim: int,
                 adapter_dim: int,
                 hidden_dim: int,
                 bias: bool,
                 add_residual: bool) -> None:
        super(TwoLayerAdapter, self).__init__()
        self.embedding_dim = embedding_dim
        self.adapter_dim = adapter_dim
        self.bias = bias
        self.add_residual = add_residual

        # 2 linear layer with activation function
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, adapter_dim)
        self.activation = nn.ReLU()
        self._add_residual = add_residual
        # if add_residual, then add residual_weight (init to 0)
        self.residual_weight = nn.Parameter(torch.zeros(1))

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # Apply 2-layer network
        embed = features["token_embeddings"]
        output1 = self.linear1(embed)
        output1 = self.activation(output1)
        output2 = self.linear2(output1)

        if self._add_residual:
            output2 = self.residual_weight * output2 + embed
        features['token_embeddings'] = output2
        return features

    def get_config_dict(self) -> Dict:
        """Get config dict."""
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_features": self.hidden_features,
            "adapter_dim": self.adapter_dim,
            "bias": self.bias,
            "add_residual": self._add_residual,
        }
