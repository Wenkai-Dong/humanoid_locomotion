from isaaclab_rl.rsl_rl import (
    RslRlPpoAlgorithmCfg,
)
class RslRlPpoAlgorithmShareCfg(RslRlPpoAlgorithmCfg):
    share_cnn_encoders: bool = True