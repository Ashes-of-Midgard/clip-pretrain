from torch import Tensor, nn
from mmyolo.models import YOLOv8CSPDarknet

from clip import AttentionPool2d


class YOLOv8Backbone(nn.Module):
    """ A CLIP image backbone using YOLOv8 csp darknet
    """
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 embed_dim: int,
                 # Attention Pool
                 num_heads: int,
                 input_resolution: int=224):
        super().__init__()
        self.input_resolution = input_resolution
        self.darknet = YOLOv8CSPDarknet(input_channels=input_channels, last_stage_out_channels=output_channels, out_indices=(4,))
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim=output_channels,
                                        num_heads=num_heads, output_dim=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.darknet(x)[0]
        x = self.attnpool(x)
        return x
    