# pytorch-snake

## Introduction

This is a simple snake game implemented in PyTorch. The snake is controlled by a neural network that takes the current state of the game as input and outputs the direction in which the snake should move. The neural network is trained using a Deep Q-Learning algorithm.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/beratcmn/pytorch-snake.git
cd pytorch-snake
pip install torch torchvision numpy matplotlib pygame
```

## Usage

To train the model, run the following command:

```bash
python agent.py
```
