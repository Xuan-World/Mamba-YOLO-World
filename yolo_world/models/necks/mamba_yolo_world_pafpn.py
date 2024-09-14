import copy
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from mmdet.utils import ConfigType, OptMultiConfig,OptConfigType

from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN

from torch.cuda.amp import autocast
@MODELS.register_module()
class MambaYOLOWorldPAFPN(YOLOv8PAFPN):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_expand: int,
                 text_emb_dim: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='MambaFusionCSPLayerWithTwoConv'),
                 text_extractor: ConfigType = dict(type='Mamba'),
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        self.guide_emb_dim = guide_expand * text_emb_dim
        self.text_emb_dim = text_emb_dim
        self.block_cfg = block_cfg
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        self.text_top_down = MODELS.build(text_extractor)
        self.text_bottom_up = MODELS.build(text_extractor)


    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_emb_dim = self.guide_emb_dim,
                 text_emb_dim = self.text_emb_dim,
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_emb_dim=self.guide_emb_dim,
                 text_emb_dim=self.text_emb_dim,
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg,
                 get_hidden_state=False
            ))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        with autocast(enabled=False):
            img_feats = [img_feat.float() for img_feat in img_feats]
            txt_feats = txt_feats.float()

            assert len(img_feats) == len(self.in_channels)
            # reduce layers
            reduce_outs = []
            for idx in range(len(self.in_channels)):
                reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

            txt_feats_res,hidden_states = self.text_top_down(txt_feats)
            txt_feats = txt_feats + txt_feats_res
            hidden_states = hidden_states.view(hidden_states.shape[0],-1,hidden_states.shape[-1]).contiguous()

            # top-down path
            img_hidden_states = []
            inner_outs = [reduce_outs[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_low = reduce_outs[idx - 1]
                upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
                if self.upsample_feats_cat_first:
                    top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
                else:
                    top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
                inner_out, img_hidden_state = self.top_down_layers[len(self.in_channels) - 1 - idx](top_down_layer_inputs, hidden_states)
                img_hidden_states.append(img_hidden_state)
                inner_outs.insert(0, inner_out)

            img_hidden_states = torch.cat(img_hidden_states, -1)
            txt_feats = torch.cat((img_hidden_states.transpose(-1,-2).contiguous(),txt_feats),dim=-2)
            txt_feats_res,hidden_states = self.text_bottom_up(txt_feats)
            txt_feats = txt_feats + txt_feats_res
            txt_feats = txt_feats[...,img_hidden_states.shape[-1]:,:]
            hidden_states = hidden_states.view(hidden_states.shape[0],-1,hidden_states.shape[-1]).contiguous()

            # bottom-up path
            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_high = inner_outs[idx + 1]
                downsample_feat = self.downsample_layers[idx](feat_low)
                out , _= self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1), hidden_states)
                outs.append(out)

            # out_layers
            results = []
            for idx in range(len(self.in_channels)):
                results.append(self.out_layers[idx](outs[idx]))

            return tuple(results),txt_feats

