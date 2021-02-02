import gym
import numpy as np

config = {'difficulty': 'hard'} 
env = gym.make('sudoku:Sudoku-v0', **config)

env.reset()

for _ in range(5000):
    env.render()
    action = env.action_space.sample()
    env.step(action)
