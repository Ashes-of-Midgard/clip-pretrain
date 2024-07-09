from typing import Tuple, Union, List
import torch
from torch import nn
from torch import Tensor

from clip import CLIP, tokenize


class CLIPZeroshotClassifier(CLIP):
    """ Implementation of a classifier in CLIP-zeroshot way
        It is finetunable with cls datasets
    """
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # cls
                 cls_categories: Union[List[str], Tuple[str, ...]]
                 ):
        super().__init__(embed_dim,
                         image_resolution,
                         vision_layers,
                         vision_width,
                         vision_patch_size,
                         context_length,
                         vocab_size,
                         transformer_width,
                         transformer_heads,
                         transformer_layers)
        self.cls_categories = cls_categories
        self.category_embeddings = nn.parameter.Parameter(self.embed_categories(self.cls_categories), requires_grad=False)

    def embed_categories(self, categories: Union[List[str], Tuple[str, ...]]) -> Tensor:
        """ Embed the categories' name to Tensors
        """
        device = next(self.parameters()).device
        category_tokens = tokenize(categories).to(device)
        with torch.no_grad():
            category_embeddings = self.encode_text(category_tokens)
            category_embeddings = category_embeddings / category_embeddings.norm(dim=1, keepdim=True)
        return category_embeddings

    def forward(self, images: Tensor):
        # extract image features
        image_features = self.encode_image(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        category_embeddings = self.category_embeddings.to(self.dtype)
        logits_per_image = logit_scale * image_features @ category_embeddings.t()

        return logits_per_image
    
    def frozen_text_backbone(self):
        self.token_embedding.requires_grad_(False)
        self.positional_embedding.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.ln_final.requires_grad_(False)
        self.text_projection.requires_grad_(False)

    @property
    def dtype(self):
        try:
            _dtype = self.visual.conv1.weight.dtype
        except:
            _dtype = torch.float32
        return _dtype
