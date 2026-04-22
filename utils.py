from random import uniform as randfloat

import gym
from ray.rllib import MultiAgentEnv
import soccer_twos

import numpy as np
from pprint import pprint

VELOCITY_SCALE = 1 / 40.0  # adjust this if you get a better estimate of max velocity


########################################################################################
# NOTE: can create a new wrapper here for another new environment
########################################################################################
# class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
#    """
#    A RLLib wrapper so our env can inherit from MultiAgentEnv.
#    """
#
#    pass


########################################################################################
# NOTE: create a environment class that rewards a player for moving towards ball
########################################################################################

class BallVelocityandPositionRewardWrapper(gym.core.Wrapper, MultiAgentEnv) :
    def __init__(self, env):
        super(BallVelocityandPositionRewardWrapper, self).__init__(env)

    def step(self, action) :
        obs, reward, done, info = self.env.step(action)
        # print(info[0])
        if len(reward) == 2: 
            reward[0] += info[0][0]['ball_info']['velocity'][0] * 0.0001 # ball velocity to right
            reward[1] += info[0][0]['ball_info']['velocity'][0] * -0.0001 # ball velocity to left
            reward[0] += 0.00001 * 1/(np.linalg.norm(info[0][0]['player_info']['position'] - info[0][0]['ball_info']['position']) + 0.01)
            reward[0] += 0.00001 * 1/(np.linalg.norm(info[0][1]['player_info']['position'] - info[0][1]['ball_info']['position']) + 0.01)
            reward[1] += 0.00001 * 1/(np.linalg.norm(info[1][0]['player_info']['position'] - info[0][0]['ball_info']['position']) + 0.01)
            reward[1] += 0.00001 * 1/(np.linalg.norm(info[1][1]['player_info']['position'] - info[0][0]['ball_info']['position']) + 0.01)
        elif len(reward) ==4 :
            reward[0] += info[0]['ball_info']['velocity'][0] * 0.001 # ball velocity to right
            reward[1] += info[1]['ball_info']['velocity'][0] * 0.001 # ball velocity to right
            reward[2] += info[2]['ball_info']['velocity'][0] * -0.001 # ball velocity to left
            reward[3] += info[3]['ball_info']['velocity'][0] * -0.001 # ball velocity to left
            reward[0] += 0.001 * 1/(np.linalg.norm(info[0]['player_info']['position'] - info[0]['ball_info']['position']) + 0.1)
            reward[1] += 0.001 * 1/(np.linalg.norm(info[1]['player_info']['position'] - info[1]['ball_info']['position']) + 0.1)
            reward[2] += 0.001 * 1/(np.linalg.norm(info[2]['player_info']['position'] - info[2]['ball_info']['position']) + 0.1)
            reward[3] += 0.001 * 1/(np.linalg.norm(info[3]['player_info']['position'] - info[3]['ball_info']['position']) + 0.1)
            # if np.linalg.norm(info[0]['player_info']['velocity']) <= 0.01 :
            #     reward[0] -= 0.0-1
            # if np.linalg.norm(info[1]['player_info']['velocity']) <= 0.01 :
            #     reward[1] -= 0.001
            # if np.linalg.norm(info[2]['player_info']['velocity']) <= 0.01 :
            #     reward[2] -= 0.001
            # if np.linalg.norm(info[3]['player_info']['velocity']) <= 0.01 :
            #     reward[3] -= 0.001
        else :
            raise ValueError("Expected the reward to be either of length two or four!")
        return obs, reward, done, info

class VelocityTowardsBallRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    def __init__(self, env):
        super(VelocityTowardsBallRewardWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if len(reward) == 2:
            velocity_towards_ball_reward = np.zeros(2)
            velocity_towards_ball_reward[0] = np.sum(
                self._compute_velocity_towards_ball(info[0][0])
                + self._compute_velocity_towards_ball(info[0][1])
            )
            velocity_towards_ball_reward[1] = np.sum(
                self._compute_velocity_towards_ball(info[1][0])
                + self._compute_velocity_towards_ball(info[1][1])
            )

            reward[0] += velocity_towards_ball_reward[0]
            reward[1] += velocity_towards_ball_reward[1]

        elif len(reward) == 4:
            velocity_towards_ball_reward = np.zeros(4)
            velocity_towards_ball_reward[0] = self._compute_velocity_towards_ball(
                info[0]
            )
            velocity_towards_ball_reward[1] = self._compute_velocity_towards_ball(
                info[1]
            )
            velocity_towards_ball_reward[2] = self._compute_velocity_towards_ball(
                info[2]
            )
            velocity_towards_ball_reward[3] = self._compute_velocity_towards_ball(
                info[3]
            )
            reward[0] += velocity_towards_ball_reward[0]
            reward[1] += velocity_towards_ball_reward[1]
            reward[2] += velocity_towards_ball_reward[2]
            reward[3] += velocity_towards_ball_reward[3]
        else:
            raise ValueError("Expected the reward to be either of length two or four!")

        # DEBUG: print rewards
        # for key, value in reward.items():
        #    print(f"reward for {key} is: {value}")
        # END DEBUG: print rewards

        return obs, reward, done, info

    def _compute_velocity_towards_ball(
        self, agent_info, velocity_scale=VELOCITY_SCALE, weight=1.0
    ):
        agent_velocity = agent_info["player_info"]["velocity"]
        agent_position = agent_info["player_info"]["position"]
        ball_position = agent_info["ball_info"]["position"]

        # compute relative direction from agent to ball
        relative_position = ball_position - agent_position
        if np.linalg.norm(relative_position) <= 1e-5:
            relative_direction = np.zeros_like(relative_position)
        else:
            relative_direction = relative_position / np.linalg.norm(relative_position)

        # reward agent for velocity in that direction but don't penalize for
        # going the wrong way
        agent_velocity_towards_ball = np.dot(agent_velocity, relative_direction)
        return np.maximum(agent_velocity_towards_ball, 0.0) * velocity_scale * weight


def create_rllib_env_with_velocity_towards_ball_reward(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return VelocityTowardsBallRewardWrapper(env)

def create_rllib_env_with_ball_velocity(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return BallVelocityandPositionRewardWrapper(env)
########################################################################################
# original code
########################################################################################


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    # env = TransitionRecorderWrapper(env)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def sample_vec(range_dict):
    return [
        randfloat(range_dict["x"][0], range_dict["x"][1]),
        randfloat(range_dict["y"][0], range_dict["y"][1]),
    ]


def sample_val(range_tpl):
    return randfloat(range_tpl[0], range_tpl[1])


def sample_pos_vel(range_dict):
    _s = {}
    if "position" in range_dict:
        _s["position"] = sample_vec(range_dict["position"])
    if "velocity" in range_dict:
        _s["velocity"] = sample_vec(range_dict["velocity"])
    return _s


def sample_player(range_dict):
    _s = sample_pos_vel(range_dict)
    if "rotation_y" in range_dict:
        _s["rotation_y"] = sample_val(range_dict["rotation_y"])
    return _s
