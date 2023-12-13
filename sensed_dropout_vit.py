import torch
import torchvision
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.vision_transformer import EncoderBlock
import torch.nn as nn

from collections import OrderedDict
from functools import partial
from typing import Any, Optional, List, Callable

from sensed_patch_dropout import SensedPatchDropout

class SensedDropoutVisionTransformer(VisionTransformer):

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        tokens: int,
        ratio: int,
        train_sampling: int,
        inference_sampling: int,
        basis: str,
        sensors: int,
        sensing_patch_size: int,
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
        
        self.train_sampling = train_sampling
        self.inference_sampling = inference_sampling

        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        self.patch_dropout = SensedPatchDropout(
            tokens=tokens,
            ratio=ratio,
            train_sampling=train_sampling, 
            inference_sampling=inference_sampling,
            basis=basis,
            sensors=sensors, 
            sensing_patch_size=sensing_patch_size
        )

        self.encoder = SensedDropoutEncoder(
            num_layers,
            num_heads,
            self.hidden_dim,
            self.mlp_dim,
            self.dropout,
            self.attention_dropout,
            self.norm_layer
        )
    
    def forward(self, x: torch.Tensor):
        sampling = self.train_sampling if self.training else self.inference_sampling
        if sampling in ['r']:
            self.patch_dropout.update_sensing_mask(x)

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = x + self.pos_embedding

        x = self.patch_dropout(x)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

class SensedDropoutEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))

def _sensed_dropout_vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    tokens: int,
    ratio: int,
    train_sampling: int,
    inference_sampling: int,
    basis: str,
    sensors: int,
    sensing_patch_size: int,
    weights,
    progress: bool,
    **kwargs: Any,
) -> SensedDropoutVisionTransformer:
    image_size = kwargs.pop("image_size", 224)

    model = SensedDropoutVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        tokens=tokens,
        ratio=ratio,
        train_sampling=train_sampling,
        inference_sampling=inference_sampling,
        basis=basis,
        sensors=sensors,
        sensing_patch_size=sensing_patch_size,
        **kwargs,
    )

    return model

def sensed_dropout_vit_b_16(*, tokens, ratio, train_sampling, inference_sampling, basis, sensors, sensing_patch_size, 
                           weights = None, progress: bool = True, **kwargs: Any) -> SensedDropoutVisionTransformer:
    return _sensed_dropout_vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        tokens=tokens,
        ratio=ratio,
        train_sampling=train_sampling,
        inference_sampling=inference_sampling,
        basis=basis,
        sensors=sensors,
        sensing_patch_size=sensing_patch_size,
        weights=weights,
        progress=progress,
        **kwargs,
    )