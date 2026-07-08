from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg
from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticRecurrentCfg(RslRlPpoActorCriticCfg):
    """Configuration for the PPO actor-critic networks."""

    rnn_type: str = "gru"



@configclass
class EmptyCfg:
    """ Empty configuration file """
