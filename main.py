from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(gamma=1,alpha=0.02)
avg_rewards, best_avg_reward = interact(env, agent)