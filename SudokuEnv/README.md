**Status:** 
- Sudoku board generation is rewritten using backtracking (standard approach) 
- Reset function is successfully reconstructed by creating copies of appropriate class variables.

**TODOS:**
- Improve Sudoku board generation for $$d = 5$$. Very slow...
- Render mode "human" needs to be integrated in the codebase using pyglet.
- Policy gradient methods need to be implemented to successfully train agent.
- Dockerfile needs fully contain all needed dependencies.
- Will be using Ray and RlLib capabilities to distribute RL training after model is successfully created.

# sudokugym

The Sudoku environment is a single-agent Gym-compliant customized containerized environment that supports puzzle creation for $$d$$-dimensions ($$d \in \{2, 3, 4, 5\}$$). The environment purposefully creates puzzles (both solvable and unsolvable) to assist strategy learning in training using policy gradient methods.

# Installation and running train.py

```bash
docker build --tag sudoku .
docker run --rm -ti -v /sudoku:/sudoku sudoku /bin/bash
python ./SudokuAgent/train.py
```
