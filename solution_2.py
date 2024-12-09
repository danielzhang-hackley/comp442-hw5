# %% [markdown]
# # Assignment 5, Problem 2
# 
# This is the starter code for Assignment 5, Problem 2.
# 
# In this assignment, you will solve increasingly challenging tasks from the [Minigrid benchmark](https://minigrid.farama.org/).

# %%
import gymnasium as gym
import minigrid
import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

# %%
def compute_score(task, policy):
  num_episodes = 10
  cur_episode  = 0

  seed_by_episode = [42, 34, 50, 1, 9, 7, 43, 56, 90, 11]
  score_by_episode = np.zeros(num_episodes)

  while cur_episode < num_episodes:

    cumulative_reward = 0
    cur_seed = seed_by_episode[cur_episode]

    observation, info = task.reset(seed=cur_seed)
    done = False

    while not done:
      action = policy(observation)
      observation, reward, terminated, truncated, info = task.step(action)
      cumulative_reward += reward

      if terminated or truncated:
        done = True
        score_by_episode[cur_episode] = cumulative_reward
        cur_episode += 1

  score_mean = round(score_by_episode.mean(), 3)
  score_std  = round(score_by_episode.std(), 3)
  score_best = round(score_by_episode.max(), 3)

  print(f"Best score: {score_best}")
  print(f"Average score: {score_mean, score_std}")

  return score_by_episode

# %% [markdown]
# ## Problem 2
# Solve the [Minigrid Unlock and Pickup](https://minigrid.farama.org/environments/minigrid/UnlockEnv/) task.
# 
# This problem is worth 10 points for COMP 442 students.
# This problem is worth 05 points for COMP 552 students.
# 
# ![](https://minigrid.farama.org/_images/UnlockPickupEnv.gif)

# %%
second_task = gym.make("MiniGrid-UnlockPickup-v0")

# %%
######## PUT YOUR CODE HERE ########
# Train an agent to solve the task
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from minigrid.envs import UnlockPickupEnv


class UnlockPickupCustomEnv(UnlockPickupEnv):
    def __init__(self, max_steps: int | None = None, verbose=0, **kwargs):
        super().__init__(max_steps, **kwargs)
        self.verbose = verbose

        self.gave_key_reward = False
        self.gave_open_reward = False
        self.gave_drop_reward = False

    def _gen_grid(self, width, height):
        super(UnlockPickupEnv, self)._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        key = self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

        self.door = door
        self.key = key

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        """
        # We don't really need this reward if we are jump start using problem 1
        # picked up an item; if it is key, alter reward
        if action == self.actions.pickup and self.carrying == self.key and (not self.gave_key_reward):
            reward = self._reward() / 5
            self.gave_key_reward = True
        """
        # We still need this if we jump start using problem 1 in order not to lose progress
        # toggled an item; if door is now unlocked and opened, alter reward
        if action == self.actions.toggle \
           and (not self.door.is_locked) and self.door.is_open \
           and (not self.gave_open_reward):
            reward = self._reward() / 3
            self.gave_open_reward = True

            if self.verbose > 0:
                print("gave agent door open reward")

        # Apparently some people are getting messed up by this, so add an extra reward
        # dropped an item; if door is unlocked by the time we drop, alter reward
        if action == self.actions.drop and (not self.door.is_locked) \
           and (not self.gave_drop_reward):
            reward = self._reward() / 5
            self.gave_drop_reward = True

            if self.verbose > 0:
                print("gave agent key drop reward")

        if terminated and self.verbose > 0:
            print("agent completed task")

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed = None, options = None):
        if self.verbose > 0:
            print()
        self.gave_open_reward = False
        self.gave_drop_reward = False
        return super().reset(seed=seed, options=options)


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


second_model = PPO.load("models/q2.pth")

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

train_task = UnlockPickupCustomEnv(verbose=1)
env = minigrid.wrappers.ImgObsWrapper(train_task)

# first_model = PPO.load("models/q1.pth")
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs)
# model.policy.load_state_dict(first_model.policy.state_dict())

# model.learn(1e6)
######## PUT YOUR CODE HERE ########
# %%
def second_policy(observation):
  ######## PUT YOUR CODE HERE ########
  action = second_model.predict(observation["image"])[0]
  ######## PUT YOUR CODE HERE ########
  return action

# %%
compute_score(task=second_task, policy=second_policy)
second_model.save("models/q2.pth")
