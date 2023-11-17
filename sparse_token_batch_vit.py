import time

import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn

from functools import partial
from typing import Any, Optional, List, Callable

from pysensors import SSPOC, SSPOR

import tokens

class SparseTokenBatchVisionTransformer(VisionTransformer):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        ps_model: SSPOR | SSPOC,
        fit_type: str,
        patch: int,
        tokens: int,
        random_tokens: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[torchvision.models.vision_transformer.ConvStemConfig]] = None,
    ):
        super().__init__(image_size, 
                         patch_size, 
                         num_layers, 
                         num_heads, 
                         hidden_dim, 
                         mlp_dim, 
                         dropout, 
                         attention_dropout, 
                         num_classes, 
                         representation_size,
                         norm_layer,
                         conv_stem_configs
        )
        
        self.token_mask = torch.zeros((self.image_size // self.patch_size, self.image_size // self.patch_size), dtype=bool)
        self.heatmap = torch.zeros((self.image_size // self.patch_size, self.image_size // self.patch_size), dtype=int)
        self.ps_model = ps_model
        self.fit_type = fit_type
        self.patch = patch
        self.tokens = tokens
        self.random_tokens = random_tokens
        self.seq_length = tokens + random_tokens + 1

        self.encoder = torchvision.models.vision_transformer.Encoder(
            self.seq_length,
            num_layers,
            num_heads,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout,
            self.attention_dropout,
            self.norm_layer
        )
    
    def update_mask(self, x: torch.Tensor, y: torch.Tensor):
        self.token_mask = torch.zeros((self.image_size // self.patch_size, self.image_size // self.patch_size), dtype=bool)
        if self.tokens != 0:
            self.token_mask = tokens.fit_mask(self.ps_model, self.fit_type, x, y, self.patch, self.tokens)
        if self.random_tokens != 0:
            zeros = (self.token_mask.ravel() == 0).argwhere().squeeze()
            new_ones = zeros[torch.randperm(len(zeros))][:self.random_tokens]
            self.token_mask.ravel()[new_ones] = True
        self.heatmap += self.token_mask
        print(self.heatmap)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        x = x[:, :, self.token_mask.ravel()]

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        return x

def _sparse_token_batch_vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    ps_model: SSPOR | SSPOC,
    fit_type: str,
    patch: int,
    tokens: int,
    random_tokens: int,
    weights,
    progress: bool,
    **kwargs: Any,
) -> SparseTokenBatchVisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = SparseTokenBatchVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        ps_model=ps_model,
        fit_type=fit_type,
        patch=patch,
        tokens=tokens,
        random_tokens=random_tokens,
        **kwargs,
    )

    return model

def sparse_token_batch_vit_b_16(*, ps_model: SSPOR | SSPOC, fit_type: str, patch: int, tokens: int, random_tokens: int,
                                weights = None, progress: bool = True, **kwargs: Any) -> SparseTokenBatchVisionTransformer:
    return _sparse_token_batch_vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        ps_model=ps_model,
        fit_type=fit_type,
        patch=patch,
        tokens=tokens,
        random_tokens=random_tokens,
        weights=weights,
        progress=progress,
        **kwargs,
    )