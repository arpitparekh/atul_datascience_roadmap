import gymnasium as gym
import numpy as np

"""
    Agent:        The learner or decision-maker.
    Environment:  Everything the agent interacts with.
    State:         A specific situation in which the agent finds itself.
    Action:       All possible moves the agent can make.
    Reward: Feedback from the environment based on the action taken.

"""

# create a game environment

env = gym.make("HalfCheetah-v4",render_mode="human")

# create a starting state
state, info = env.reset()


for i in range(1,1000):
  env.render()
  action = env.action_space.sample()
  result = env.step(action)

  state, reward, done,truncated, info = result

  if done or truncated:
    state, info = env.reset()

  print(state, reward, done,truncated, info)

env.close()
