SSM_D_STATE = 64
SSM_DT_RANK = "auto"
SSM_RATIO = 1.0
SSM_CONV = 3
MLP_RATIO = 2.0
NORM_LAYER = "ln2d"
SSM_DROP_RATE = 0.0
MLP_DROP_RATE = 0.0
REPEAT_SS2D = 1
channel_first = False #VMamba2特点，必须False
SSM_FORWARDTYPE = "m0"
SSM_CONV_BIAS = False,

TEXT_HIDDEN_DIM = 64
TEXT_CHANNELS = 512
TEXT_EXPAND = 1
REPEAT_SSM = 1

ss2d_config = dict( type='TextGuidedSS2D2',
                    d_state=SSM_D_STATE,
                    ssm_ratio=SSM_RATIO,
                    dt_rank=SSM_DT_RANK,
                    # ==========================
                    d_conv=SSM_CONV,
                    conv_bias=SSM_CONV_BIAS,
                    # ==========================
                    dropout=SSM_DROP_RATE,
                    channel_first = channel_first,
                    # ==========================
                    guide_hidden_dim = TEXT_HIDDEN_DIM,
                    forward_type=SSM_FORWARDTYPE,
                   )

vss_cfg =dict(type='TextGuidedODSSBlock2',
              ss2d_config = ss2d_config,
              norm_layer=NORM_LAYER,
              mlp_ratio=MLP_RATIO,
              mlp_drop_rate=MLP_DROP_RATE,
              n= REPEAT_SS2D
)

mamba_cfg = dict(type='Mamba2Simple',
                  # This module uses roughly 3 * expand * d_model^2 parameters
                  d_model=TEXT_CHANNELS,  # Model dimension d_model
                  d_state=TEXT_HIDDEN_DIM,  # SSM state expansion factor, typically 64 or 128
                  d_conv=SSM_CONV,  # Local convolution width
                  expand=TEXT_EXPAND,  # Block expansion factor
                  headdim = 64,
                  )