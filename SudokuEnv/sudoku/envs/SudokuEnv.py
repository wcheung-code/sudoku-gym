import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
from itertools import product
from collections import defaultdict
from ast import literal_eval
import pyglet

# Utility Functions:
def baseline(row, col, dim):
    return (dim * (row % dim) + row // dim + col) % (dim**2)

def shuffle(s):
    np.random.shuffle(s)
    return s

def convert(g):
    return ''.join(map(lambda x: ''.join(map(str, x)), g.tolist()))

def get_possibilities(puzzle, dim):
    constraints = list()
    square = dict()
    counter = 0
    puzzle_dict_format = dict()
    possibilities = defaultdict(set)

    for i, row in enumerate(puzzle):
        for j, value in enumerate(row):
            puzzle_dict_format[(i, j)] = value

    for index in np.arange(dim**2):
        constraints.append(set([(index, i) for i in np.arange(dim**2)]))
        constraints.append(set([(i, index) for i in np.arange(dim**2)]))


    for index in dim*np.arange(dim):
        for tulip in [(index, i) for i in dim*np.arange(dim)]:
            square[counter] = tulip
            counter += 1

    for row, col in square.values():
        row_ids = [row + i for i in np.arange(dim)]
        col_ids = [col + j for j in np.arange(dim)]
        constraints.append(set(list(product(row_ids, col_ids))))

    for k, v in puzzle_dict_format.items():
        for checker in [s for s in constraints if k in s]:
            if not v: ## v = 0
                possible_values = set(map(lambda x: puzzle_dict_format[x], checker - set((k, v)))) - {v}
                possibilities[k] |= possible_values
    possibilities = {k: set(np.arange(1, dim**2 + 1)) - v for k, v in possibilities.items()}

    return possibilities

def given_value(value):
    return spaces.Box(low=value, high=value, shape=(1, ), dtype=np.int)

def action_space_i(dim):
    from math import gcd
    a = range(1, dim**2 + 1)
    lcm = a[0]
    for i in a[1:]:
        lcm = lcm*i//gcd(lcm, i)
    return lcm

def build(line, dim):
    return line[0] + line[6:11].join([line[1:6]*(dim-1)]*dim) + line[11:16]

class ConfigClass:
    def __init__(self, dim):
        self.dim = dim
        self.config = dict()
        for variable in ['rows', 'cols', 'nums']:
            if variable != 'nums':
                self.config[variable] = np.array([g*self.dim + i for g, i in product(shuffle(np.arange(self.dim)), repeat = 2)])
            else:
                self.config[variable] = shuffle(np.arange(1, self.dim**2 + 1))

class SudokuSolutionClass(ConfigClass):
    def __init__(self, dim):
        super().__init__(dim)
        rows, cols, nums = self.config['rows'], self.config['cols'], self.config['nums']
        self.solution = np.array([nums[baseline(row, col, self.dim)] for row, col in product(rows, cols)])
        self.solution = self.solution.reshape((self.dim**2, self.dim**2))

## null_percentage is encouraged to be less than 0.75
class SudokuProblemClass(SudokuSolutionClass):
    def __init__(self, dim, null_percentage):
        super().__init__(dim)
        self.solution = SudokuSolutionClass(self.dim).solution
        self.puzzle = self.solution
        self.null_percentage = null_percentage
        null_n = int(null_percentage * dim**4)
        null_indices = shuffle(list(product(np.arange(dim**2), repeat = 2)))[:null_n]
        for i, j in null_indices:
            self.puzzle[i, j] = 0
        self.possibilities = get_possibilities(self.puzzle, self.dim)

class SudokuProblemSpace(gym.Space):
    def __init__(self, dim, null_percentage):
        assert isinstance(null_percentage, float) and null_percentage > 0 and null_percentage <= 1
        assert isinstance(dim, int) and dim in [2, 3, 4, 5]
        self.dim, self.null_percentage = dim, null_percentage
        sudoku_problem_class = SudokuProblemClass(self.dim, self.null_percentage)
        self.problem = sudoku_problem_class.puzzle
        self.possibilities = sudoku_problem_class.possibilities
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        sudoku_problem_class = SudokuProblemClass(self.dim, self.null_percentage)
        self.problem = sudoku_problem_class.puzzle
        self.possibilities = sudoku_problem_class.possibilities
        return self.problem
    
    def __repr__(self):
        return "SudokuProblemSpace({}, {})".format(self.dim, self.null_percentage)

class SudokuEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, dim, null_percentage):
        self.dim, self.null_percentage = dim, null_percentage
        self.changed = False
        self.board = board = {
            0: '╔════╤════╦════╗',
            1: '║ .. │ .. ║ .. ║',
            2: '╟────┼────╫────╢',
            3: '╠════╪════╬════╣',
            4: '╚════╧════╩════╝'
        }
        self._build_spaces()

        self.mapping = dict()
        for i in range(1, self.dim**6 + 1):
            x = (i-1)//self.dim**4
            y = (i - self.dim**4*x - 1)//(self.dim**2)
            value = self.dim**2 if not i%(self.dim**2) else i%(self.dim**2)
            self.mapping[i-1] = (value, (x, y))

    def step(self, action):
        target_value, target_cell = self.mapping[action]
        if target_cell in self.possibilities.keys():
            if target_value in self.possibilities[target_cell]:
                self.problem[target_cell] = target_value
                self.changed = True
                reward = 1 if np.all(self.problem != 0) else 0
                done = True if np.all(self.problem != 0) else False
            else:
                reward = -1 if len(self.possibilities[target_cell]) == 0 else 0
                done = True if len(self.possibilities[target_cell]) == 0 else False
        else:
            reward, done = 0, False
        if self.changed:
            self.possibilities = get_possibilities(self.problem, self.dim)
            self.changed = False
        return self.problem, reward, done

    def reset(self):
        return self.observation_space.sample()

    def render(self, mode='ansi'):
        return self._build_ansi_board(self.observation_space.problem)

    def close(self):
        pass

    def _build_spaces(self):
        self.observation_space = SudokuProblemSpace(self.dim, self.null_percentage)
        self.action_space = spaces.Discrete(self.dim**6)
        self.problem = self.observation_space.problem
        self.possibilities = self.observation_space.possibilities

    def _build_ansi_board(self, problem):
        puzzle = [build(self.board[0], self.dim)]
        for i, row in enumerate(problem):
            row = np.where(np.invert(row.astype(bool)), '  ', np.char.zfill(row.astype(str), 2))
            _, a, b, c, d = tuple(map(lambda x: build(x, self.dim), self.board.values()))
            puzzle.append(''.join(sum(zip(np.concatenate(([''], row)), a.split('..')), ())) )
            puzzle.append([b, c, d][((i+1)%(self.dim**2)==0)+((i+1)%(self.dim)==0)])
        return '\n'.join(puzzle)