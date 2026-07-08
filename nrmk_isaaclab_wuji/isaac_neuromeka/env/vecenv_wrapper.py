# Modified from Orbit's implementation


"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to NRMK-RL vectorized environment.

The following example shows how to wrap an environment for NRMK-RL:

.. code-block:: python

    from isaaclab_tasks.utils.wrappers.nrmk_rl import NrmkRlVecEnvWrapper

    env = NrmkRlVecEnvWrapper(env)

"""
import pdb  # noqa:F401

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv


class NrmkRlVecEnvWrapper:
    """Wraps around Orbit environment for NRMK-RL library"""

    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the NRMK-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv):
            raise ValueError(f"The environment must be inherited from ManagerBasedRLEnv. Environment type: {type(env)}")
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self.obs_dict = {}
        # self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        # # -- privileged observations
        # if "critic" in self.unwrapped.observation_manager.group_obs_dim:
        #     self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        # else:
        #     self.num_privileged_obs = 0
        # # reset at the start since the NRMK-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    @property
    def num_cost_terms(self) -> int:
        """Returns the number of cost terms in the environment."""
        return self.env.cost_manager.num_cost_terms

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        return self.get_flat_observation(self.obs_dict), {"observations": self.obs_dict}

    def get_flat_observation(self, obs_dict) -> torch.Tensor:

        # cases
        ## For training: actor_obs_list exists and each obs is a tensor
        ## For testing or some cases: some obs are dict and some are tensor
        ## At the end, we want to concat all the tensors

        if hasattr(self.cfg, "actor_obs_list"):

            obs_list_to_concat = []
            for key in self.cfg.actor_obs_list:
                obs_tensor = None
                # if obs_dict[key] is a dict, then concatenate the values
                if isinstance(obs_dict[key], dict):
                    # obs_tensor = torch.cat(list(obs_dict[key].values()), dim=-1)
                    obs_tensor = torch.cat(
                        [value.reshape(self.num_envs, -1) for value in obs_dict[key].values()], dim=-1
                    )  # need to check if this works
                else:
                    obs_tensor = obs_dict[key].reshape(self.num_envs, -1)
                obs_list_to_concat.append(obs_tensor)
            return torch.cat(obs_list_to_concat, dim=-1)
        else:
            if isinstance(obs_dict["policy"], dict):
                return torch.cat(list(obs_dict["policy"].values()), dim=-1)
            else:
                return obs_dict["policy"]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in NRMK-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, extras = self.env.reset()
        self.obs_dict = obs_dict
        obs = self.get_flat_observation(obs_dict)
        extras["observations"] = obs_dict
        return obs, extras

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with NRMK-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        self.obs_dict = obs_dict
        obs = self.get_flat_observation(obs_dict)

        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    # # Used for NRMK-RL collision avoidance pretraining
    # def set_estimation(self, estimation: torch.Tensor):
    #     self.env.set_estimation(estimation)
