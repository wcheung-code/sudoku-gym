# Sudoku Gym Environment

The Sudoku environment is a single-agent Gym-compliant custom environment that relies on the [Sugoku API](https://github.com/bertoort/sugoku) for puzzle generation and the SAT solver in [z3](https://github.com/Z3Prover/z3) to define rewards and ensure uniqueness of solutions. The environment is built to support training of reinforcement learning algorithms aimed to replicate known strategies used in Sudoku solving competitions. This is a work in progress.

## Setting Up the Environment

```bash
docker build --tag sudoku .
docker run --rm -ti -v /sudoku:/sudoku sudoku /bin/bash
```

## Running `demo.py`:

```bash
python ./SudokuAgent/demo.py
```