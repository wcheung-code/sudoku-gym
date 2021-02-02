# Use an official Python runtime as a parent image
FROM python:3.7
LABEL maintainer="Wilson Cheung"

# Install Requirements
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y --no-install-recommends \
        curl wget \
        sudo \
        vim \
        unzip

COPY ./requirements.txt /Sudoku/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /Sudoku/requirements.txt

# Copy the ./Academy directory contents into the container at /Academy/
COPY ./SudokuEnv /Sudoku/SudokuEnv/
COPY ./SudokuAgent /Sudoku/SudokuAgent/

# Set the working directory to /Academy/
WORKDIR /Sudoku/
RUN pip install ray[rllib]
RUN pip install -e SudokuEnv

# Run FastAPI server
# CMD [ "python", "./AcademyAPI.py" ]
# CMD ["uvicorn", "AcademyAPI:app", "--host", "0.0.0.0", "--port", "4000"]