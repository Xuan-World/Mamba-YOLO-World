# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv
    )
from .mamba2_yolo_bricks import (
    MambaFusionCSPLayerWithTwoConv2,
    TextGuidedODSSBlock2,
    TextGuidedSS2D2
)
from .mamba2_simple import Mamba2Simple

from .mamba2_yolo_bricks_AB_noTextGuide import (
    MambaFusionCSPLayerWithTwoConv2_noGuide,
    TextGuidedODSSBlock2_noGuide,
    TextGuidedSS2D2_noGuide
)

__all__ = ['CSPLayerWithTwoConv',
           'MaxSigmoidAttnBlock',
           'MaxSigmoidCSPLayerWithTwoConv',
           'RepConvMaxSigmoidCSPLayerWithTwoConv',
           'RepMaxSigmoidCSPLayerWithTwoConv',
           'ImagePoolingAttentionModule',
           'MambaFusionCSPLayerWithTwoConv2',
           'TextGuidedODSSBlock2',
           'TextGuidedSS2D2',
           'Mamba2Simple',
           'MambaFusionCSPLayerWithTwoConv2_noGuide',
           'TextGuidedODSSBlock2_noGuide',
           'TextGuidedSS2D2_noGuide',]
