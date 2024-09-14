# Copyright (c) Tencent Inc. All rights reserved.
from functools import partial
from typing import List, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv
from yolo_world.models.layers.VMamba.models.vmamba import SS2D_mamba2,LayerNorm2d,XSSBlock
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from .VMamba.models.csm_triton2 import cross_scan_fn, cross_merge_fn
_NORMLAYERS = dict(
    ln=nn.LayerNorm,
    ln2d=LayerNorm2d,
    bn=nn.BatchNorm2d,
)
@MODELS.register_module()
class TextGuidedSS2D2(SS2D_mamba2):
    def __init__(self,**kwargs,):
        super().__init__(**kwargs)
        self.guide_in_proj = nn.Linear(self.d_model, self.d_inner, bias=self.bias) if self.d_model!=self.d_inner else nn.Identity()
        #guide proj ======================================
        self.guide_hidden_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.guide_hidden_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.guide_hidden_proj], dim=0))
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=True, dstate=self.d_state),
        )
        self.forward_core = FORWARD_TYPES.get(self.forward_type, None)
        self.forward = self.forward_mm
    def forward_corem0(
            self,
            x: torch.Tensor = None,
            guide: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            chunk_size=64,
            dstate=64,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):

        assert scan_mode in ["unidi", "bidi", "cross2d"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _,guide_hidden_state_dim,guide_emb_dim = guide.shape

        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=_scan_mode,force_torch=scan_force_torch)  # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        guide_dbl = torch.einsum("b l k d, k c d -> b l k c", guide.unsqueeze(2).repeat(1, 1, 4, 1), self.guide_hidden_proj_weight)
        x_dbl = torch.cat((guide_dbl, x_dbl), dim=1)

        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)

        xs =  torch.cat((guide.unsqueeze(2).repeat(1, 1, 4, 1),xs),dim=1)
        xs = xs.contiguous().view(B, L+guide_hidden_state_dim, KR, D)
        dts = dts.contiguous().view(B, L+guide_hidden_state_dim, KR)
        Bs = Bs.contiguous().view(B, L+guide_hidden_state_dim, K, N)
        Cs = Cs.contiguous().view(B, L+guide_hidden_state_dim, K, N)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        ys, final_state = mamba_chunk_scan_combined(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
        )
        ys = ys[:,guide_hidden_state_dim:,...]
        final_state = final_state.view(B, K * RD, N)

        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False,
                                         scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1))

        return y.to(x.dtype),final_state.detach()

    def forward_mm(self, x: torch.Tensor, guide: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        guide = self.guide_in_proj(guide)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        guide = self.act(guide)
        y,final_state= self.forward_core(x,guide)
        y = self.out_act(y)
        final_state = self.out_act(final_state)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out,final_state

@MODELS.register_module()
class TextGuidedODSSBlock2(XSSBlock):
    def __init__(self ,ss2d_config: OptConfigType = None, **kwargs,):
        n = kwargs.get("n")
        kwargs.update({"n": 0})
        super().__init__(**kwargs)
        ss2d_config.update(d_model=self.hidden_dim)
        self.ss2d = nn.ModuleList([MODELS.build(ss2d_config) for _ in range(n)])
    def forward(self, input, guide):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        normed_X1 = self.norm(X1).permute(0,2,3,1).contiguous()
        guide = guide.permute(0,2,1).contiguous()
        image_hidden_state = []
        for ss2d_layer in self.ss2d:
            normed_X1,ihs = ss2d_layer(normed_X1, guide)
            image_hidden_state.append(ihs)
        image_hidden_state = torch.cat(image_hidden_state, dim=-1)
        normed_X1 = normed_X1.permute(0,3,1,2).contiguous()
        output = input + self.drop_path(normed_X1)
        # ===================
        if self.mlp_branch:
            output = output + self.drop_path(self.mlp(self.norm2(output)))
        return output,image_hidden_state



@MODELS.register_module()
class MambaFusionCSPLayerWithTwoConv2(CSPLayerWithTwoConv):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_emb_dim: int,
            text_emb_dim:int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            vss_cfg: OptConfigType = None,
            get_hidden_state = True
                ) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        self.get_hidden_state = get_hidden_state
        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        norm_layer: nn.Module = _NORMLAYERS.get(vss_cfg.norm_layer.lower(), None)

        vss_cfg.update(in_channels= self.mid_channels)
        vss_cfg.update(hidden_dim= self.mid_channels)
        vss_cfg.update(norm_layer=norm_layer)
        k_group = 4
        ssm_ratio = vss_cfg.get("ss2d_config").get("ssm_ratio")
        self.textGuidedVSSBlock = MODELS.build(vss_cfg)
        self.guide2img = nn.Sequential(nn.Conv1d(guide_emb_dim, self.mid_channels,kernel_size=1),
                                       nn.BatchNorm1d(self.mid_channels),
                                       nn.SiLU())
        if get_hidden_state:
            self.img2guide = nn.Sequential(nn.Conv1d(int(self.mid_channels*k_group*ssm_ratio), text_emb_dim, kernel_size=1, groups=k_group),
                                           nn.BatchNorm1d(text_emb_dim),
                                           nn.SiLU())
    def forward(self, x: Tensor, guide: Tensor) -> tuple[Any, Any]:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        guide = self.guide2img(guide)
        text_guided_x, x_hidden_state =  self.textGuidedVSSBlock(x_main[-1], guide)
        x_main.append(text_guided_x)
        if self.get_hidden_state:
            x_hidden_state = self.img2guide(x_hidden_state)
        else:
            x_hidden_state = None
        return self.final_conv(torch.cat(x_main, 1)),x_hidden_state
