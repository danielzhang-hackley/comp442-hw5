# %% [markdown]
# # Assignment 5, Problem 1
# 
# This is the starter code for Assignment 5, Problem 1.
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
# ## Problem 1
# Solve the [Minigrid Unlock](https://minigrid.farama.org/environments/minigrid/UnlockEnv/) task.
# 
# This problem is worth 5 points.
# 
# ![](https://minigrid.farama.org/_images/UnlockEnv.gif)

# %%
first_task = gym.make("MiniGrid-Unlock-v0")

# %%
######## PUT YOUR CODE HERE ########
# Train an agent to solve the task
import torch
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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


first_model = PPO.load("models/q1.pth")

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = minigrid.wrappers.ImgObsWrapper(first_task)

# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(5e5)
######## PUT YOUR CODE HERE ########
# %%
def first_policy(observation):
  ######## PUT YOUR CODE HERE ########
  action = first_model.predict(observation["image"])[0]
  ######## PUT YOUR CODE HERE ########
  return action

# %%
compute_score(task=first_task, policy=first_policy)
first_model.save("models/q1.pth")
