**Status:** Reset function in environment needs to be fixed: may possibly require re-engineering observation and action spaces. Render mode "human" needs to be integrated in the codebase using pyglet. Policy gradient methods need to be implemented to successfully train agent. Dockerfile needs to be fully contained.

# sudokugym

The Sudoku environment is a single-agent Gym-compliant customized containerized environment that supports puzzle creation for $d$-dimensions ($d$ \in \{2, 3, 4, 5\}). 


# Installation and running train.py

```bash
docker build --tag sudoku .
docker run --rm -ti -v /sudoku:/sudoku sudoku /bin/bash
python ./SudokuAgent/train.py
```
