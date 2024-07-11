from torch import Tensor, nn
from mmyolo.models import YOLOv8CSPDarknet
from typing import Tuple, Union, List
from mmdet.utils import ConfigType, OptMultiConfig
import math

from clip import AttentionPool2d


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


class YOLOv8Backbone(nn.Module):
    """ A CLIP image backbone using YOLOv8 csp darknet
    """
    def __init__(self,
                 # attention pool related
                 num_heads: int,
                 input_resolution: int,
                 embed_dim: int,
                 # csp darknet related
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (4,),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.darknet = YOLOv8CSPDarknet(arch=arch,
                                        last_stage_out_channels=last_stage_out_channels,
                                        plugins=plugins,
                                        deepen_factor=deepen_factor,
                                        widen_factor=widen_factor,
                                        input_channels=input_channels,
                                        out_indices=out_indices,
                                        frozen_stages=frozen_stages,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg,
                                        norm_eval=norm_eval,
                                        init_cfg=init_cfg)
        darknet_out_channels = make_divisible(last_stage_out_channels, widen_factor)
        self.attnpool = AttentionPool2d(input_resolution // 32,
                                        darknet_out_channels,
                                        num_heads,
                                        embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.darknet(x)[0]
        x = self.attnpool(x)
        return x
    