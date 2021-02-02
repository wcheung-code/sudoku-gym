import gym
from gym import spaces
import numpy as np
import requests
from ast import literal_eval
from z3 import Solver, Int, Or, Distinct, sat
from itertools import product

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class SudokuObservationSpace(gym.Space):

    def __init__(self, difficulty):
        assert difficulty in ['easy', 'medium', 'hard', 'random']
        self.difficulty = difficulty
        self.url = 'https://sugoku.herokuapp.com/board?difficulty={}'
        self._generate_problem()
        gym.Space.__init__(self, (), np.int64)
    
    def sample(self):
        self._generate_problem()
        return self.problem
    
    def _generate_problem(self):
        r = requests.get(self.url.format(self.difficulty), verify=False)
        self.problem = np.array(literal_eval(r.text)['board'])
    
    def _shape(self):
        return self.problem.shape
    
class SudokuSolver:

    def __init__(self):
        self.rows, self.cols = '012345678', '012345678'
        self.positions = list(map(lambda a: a[0] + a[1], product(self.rows, self.cols)))
        self.symbols = {pos: Int(pos) for pos in self.positions}
        self.solver = None      
        
    def solve(self, problem):

        self.set_conditions(problem)

        if self.solver.check() != sat:
            raise Exception("No solution.")

        model = self.solver.model()
        solution = {pos: int(model.evaluate(s).as_string()) for pos, s in self.symbols.items()}
        return np.array(list(solution.values())).reshape(9,9)
    
    def set_conditions(self, problem):
        self._initialize_solver()
        self._add_value_conditions()
        self._add_row_conditions()
        self._add_col_conditions()
        self._add_block_conditions()
        self._import_problem(problem)

    def _initialize_solver(self):
        self.solver = Solver()
    
    def _add_value_conditions(self):
        for symbol in self.symbols.values():
            self.solver.add(Or([symbol == i for i in range(1, 10)]))
    
    def _add_row_conditions(self):
        for row in self.rows:
            self.solver.add(Distinct([self.symbols[row + col] for col in self.cols]))

    def _add_col_conditions(self):
        for col in self.cols:
            self.solver.add(Distinct([self.symbols[row + col] for row in self.rows]))
        
    def _add_block_conditions(self):
        for i, j in product(range(3), repeat=2):
            blocks = [self.symbols[self.rows[m + 3*i] + self.cols[n + 3*j]] for m, n in product(range(3), repeat=2)]
            self.solver.add(Distinct(blocks))
            
    def _import_problem(self, problem):
        for row, col, value in zip(*np.nonzero(problem) + (problem[np.nonzero(problem)], )):
            pos = str(row) + str(col)
            self.solver.add(self.symbols[pos] == str(value))
            
class ActionTransformer:

    def __init__(self):
        return
    
    def _decode_action(self, action):
        ### action from action_space to (row, column, value) [row, col, value all int]
        return tuple(map(int, str(int(np.base_repr(action, base=9)) + 1).zfill(3)))

    def _encode_action(self, action_tuple):
        ### (row, column, value) [row, col, value all int] to action in action_space
        row, col, value = action_tuple
        return 81*row + 9*col + value - 1
    

'''
class SudokuActionSpace(gym.Space, ActionTransformer):

    def __init__(self):
        self.num_actions = 729
        super().__init__()
    
    def sample(self, problem):
        actions = np.arange(self.num_actions)
        return np.random.choice(actions, p = self._action_probabilities(problem))

    def _action_probabilities(self, problem):
        prob_weights = np.zeros(self.num_actions) 
        for coord in product(range(9), repeat=2):
            if not _get_options(coord, problem):
                continue
            freq = len( _get_options(coord, problem))
            for value in _get_options(coord, problem):
                encoded_action = self._encode_action(coord + (value,))
                prob_weights[encoded_action] = 1/freq
        return self._normalize(prob_weights)

    def _normalize(self, probs):
        prob_factor = 1 / sum(probs)
        return np.array([prob_factor * p for p in probs])
    
    def _get_options(self, coord, problem):
        options = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        if problem[coord]:
            return set()
        for indices in _get_coord_checks(coord):
            if not problem[indices]:
                continue
            options -= set([problem[indices]])
        return options

    def _get_coord_checks(self, coord):
        rows, cols = '012345678', '012345678'
        lookup = set()
        for row in rows:
            check = set([(int(row), int(col)) for col in cols])
            if coord in check:
                lookup |= check
        for col in cols:
            check = set([(int(row), int(col)) for row in rows])
            if coord in check:
                lookup |= check
        for i, j in product(range(3), repeat=2):
            check = set([(int(rows[m + 3*i]), int(cols[n + 3*j])) for m, n in product(range(3), repeat=2)])
            if coord in check:
                lookup |= check
        lookup -= set([coord])
        return lookup
    
    def _shape(self):
        return (self.num_actions, 1)
'''

class SudokuEnv(gym.Env, ActionTransformer):

    metadata = {'render.modes': ['human']}

    def __init__(self, difficulty):
        super(SudokuEnv, self).__init__()
        self.difficulty = difficulty
        self.solver = SudokuSolver()
        self.observation_space = SudokuObservationSpace(self.difficulty)
        self.action_space = spaces.Discrete(729)
        self.problem = None
        self.solution = None

    def reset(self):
        self.problem = self.observation_space.sample()
        self.solution = self.solver.solve(self.problem)
        return self.problem

    def step(self, action):
        row, col, value = self._decode_action(action)
        match = (self.solution[(row, col)] == value)
        complete = np.all(self.solution == self.problem)        
        if match and self.problem[(row, col)] == 0:
            self.problem[(row, col)] = value
            return self.problem, 1 if complete else 0, complete
        else:
            return self.problem, -1, complete
    
    def render(self, mode='human', close=False):
        print(self.problem)
        return self.problem