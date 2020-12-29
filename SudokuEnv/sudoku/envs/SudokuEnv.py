import gym
from gym import error, spaces, utils
import numpy as np
from itertools import product
from collections import defaultdict

# Utility Functions:
def baseline(row, col, dim):
    return (dim * (row % dim) + row // dim + col) % (dim**2)

def shuffle(s):
    np.random.shuffle(s)
    return s

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
        
class SudokuProblemClass(SudokuSolutionClass):
    def __init__(self, dim):
        super().__init__(dim)
        self.solution_class = SudokuSolutionClass(self.dim)
        self.solution = self.solution_class.solution.copy()
        self.problem = self.solution_class.solution.copy()
        self.attempted = self.solution_class.solution.copy()
        self.possibilities = None
        
    def _find_empty_location(self, arr, l):
        for row, col in product(range(self.dim**2), repeat = 2):
            if arr[row][col] == 0:
                l[0], l[1] = row, col
                return True
        return False
    
    def _used_in_row(self, arr, row, num):
        for i in range(self.dim**2):
            if arr[row][i] == num:
                return True
        return False
    
    def _used_in_col(self, arr, col, num):
        for i in range(self.dim**2):
            if arr[i][col] == num:
                return True
        return False
    
    def _used_in_box(self, arr, row, col, num):
        for i, j in product(range(self.dim), repeat = 2):
            if arr[i + row][j + col] == num:
                return True
        return False
    
    def _check_location_is_safe(self, arr, row, col, num):
        return (not self._used_in_row(arr, row, num) and
               not self._used_in_col(arr, col, num) and
               not self._used_in_box(arr, row - row % self.dim, 
                               col - col % self.dim, num))

    def solve_sudoku(self, arr):
        cache = [0, 0]
        if not self._find_empty_location(arr, cache):
            return True
        row, col = tuple(cache)
        for num in range(1, self.dim**2 + 1):
            if self._check_location_is_safe(arr, row, col, num):
                arr[row][col]= num 
                if self.solve_sudoku(arr):
                    return True
                arr[row][col] = 0
        return False
    
    def _reset(self):
        self.solution_class = SudokuSolutionClass(self.dim)
        self.solution = self.solution_class.solution
        self.problem = self.solution_class.solution
        self.attempted = self.solution_class.solution.copy()
        return True
    
    def _add_zero(self):
        row = np.random.randint(0, self.dim**2)
        col = np.random.randint(0, self.dim**2)
        while True:
            if self.problem[row][col]:
                self.problem[row][col] = 0
                self.attempted[row][col] = 0
                return True
            else:
                row = np.random.randint(0, self.dim**2)
                col = np.random.randint(0, self.dim**2)
                
    def _compare_solution(self):
        self.solve_sudoku(self.attempted)
        if np.all(self.attempted == self.solution):
            self.attempted = self.problem.copy()
            return True
        self.attempted = self.problem.copy()
        return False
    
    def generate_problem(self):
        while True:
            previous_problem = self.problem.copy()
            self._add_zero()
            if self._compare_solution():
                continue
            else:
                self.possibilities = get_possibilities(previous_problem, self.dim)
                return previous_problem

class SudokuProblemSpace(gym.Space):
    def __init__(self, dim):
        assert isinstance(dim, int) and dim in [2, 3, 4, 5]
        self.dim = dim
        self.sample()
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        self._reconstruct()
        self.problem = self.intermediate.problem.copy()
        self.possibilities = self.intermediate.possibilities.copy()
        return self.problem
    
    def _reconstruct(self):
        self.intermediate = SudokuProblemClass(self.dim)
        self.intermediate.generate_problem()
    
    def __repr__(self):
        return "SudokuProblemSpace({})".format(self.dim)

class SudokuEnv(gym.Env):
    metadata = {'render.modes': ['ansi']}

    def __init__(self, dim):
        self.dim = dim
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
        self.problem = self.observation_space.sample()
        self.possibilities = get_possibilities(self.problem, self.dim)
        return self.problem

    def render(self, mode='ansi'):
        return self._build_ansi_board(self.problem)

    def close(self):
        pass

    def _build_spaces(self):
        self.observation_space = SudokuProblemSpace(self.dim)
        self.action_space = spaces.Discrete(self.dim**6)
        self.problem = self.observation_space.problem.copy()
        self.possibilities = self.observation_space.possibilities.copy()

    def _build_ansi_board(self, problem):
        puzzle = [build(self.board[0], self.dim)]
        for i, row in enumerate(problem):
            row = np.where(np.invert(row.astype(bool)), '  ', np.char.zfill(row.astype(str), 2))
            _, a, b, c, d = tuple(map(lambda x: build(x, self.dim), self.board.values()))
            puzzle.append(''.join(sum(zip(np.concatenate(([''], row)), a.split('..')), ())) )
            puzzle.append([b, c, d][((i+1)%(self.dim**2)==0)+((i+1)%(self.dim)==0)])
        return '\n'.join(puzzle)