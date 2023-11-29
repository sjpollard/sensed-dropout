import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn

from functools import partial
from typing import Any, Optional, List, Callable

class SparseTokenVisionTransformer(VisionTransformer):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        token_mask: torch.Tensor,
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
        
        self.token_mask = token_mask
        self.seq_length = torch.sum(token_mask, dtype=int).item() + 1

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

        x = x[:, :, self.token_mask]

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        
        return x

def _sparse_token_vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    token_mask: torch.Tensor,
    weights,
    progress: bool,
    **kwargs: Any,
) -> SparseTokenVisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = SparseTokenVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        token_mask=token_mask,
        **kwargs,
    )

    return model

def sparse_token_vit_b_16(*, token_mask: torch.Tensor, weights = None, progress: bool = True, **kwargs: Any) -> SparseTokenVisionTransformer:
    return _sparse_token_vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        token_mask=token_mask,
        weights=weights,
        progress=progress,
        **kwargs,
    )