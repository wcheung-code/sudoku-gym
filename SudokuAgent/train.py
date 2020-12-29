import gym
import numpy as np

def convert(g):
    return ''.join(map(lambda x: ''.join(map(str, x)), g.tolist()))

# Create the environment
config = {'dim': 3}

env = gym.make('sudoku:Sudoku-v0', **config)

### TO BE CONTINUED



