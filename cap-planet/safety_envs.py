from typing import NamedTuple, Any
import numpy as np

import gym
import dm_env
from dm_env import StepType
from dm_control import suite

CONSTRAINED_GYM_ENVS = ['CarRacingSkiddingConstrained-v0']
CONSTRAINED_CONTROL_SUITE_ENVS = ['cartpole-swingup-constrained']

class TimeStepWithCost(NamedTuple):
  step_type: Any
  reward: Any
  discount: Any
  observation: Any
  cost: Any

  def first(self) -> bool:
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    return self.step_type == StepType.MID

  def last(self) -> bool:
    return self.step_type == StepType.LAST

class CartpoleConstrainedWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
    
    def step(self, action):
        state = self._env.step(action)
        pos = state.observation["position"][1:]
        angle = np.degrees(np.arctan2(*pos))
        cost = 1 if 20 < angle and angle < 50 else 0
        return TimeStepWithCost(*state, cost)

    def reset(self):
        return self._env.reset()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    
def load_suite_env(env_name, seed):
    spec = env_name.split('-')
    domain, task = spec[:2]
    is_safety_constrained = len(spec) > 2 and spec[2] == "constrained"

    env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})

    if is_safety_constrained and domain == "cartpole":
        env = CartpoleConstrainedWrapper(env)
    
    return env

class SkiddingConstrainedCarRacing(gym.Wrapper):
    def __init__(self):
        env = gym.make("CarRacing-v0")
        super(SkiddingConstrainedCarRacing, self).__init__(env)
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        cost = 0
        car = self.env.car
        for wheel in car.wheels:
            if wheel.skid_start is not None or wheel.skid_particle is not None:
                cost = 1
        info["cost"] = cost
        return obs, rew, done, info

gym.register("CarRacingSkiddingConstrained-v0", entry_point="safety_envs:SkiddingConstrainedCarRacing")