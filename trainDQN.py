from minesweeper import Board
# from minesweeper_env import *
from DQNsetup import *
import numpy as np
import pandas as pd
import math

# Learning settings
BATCH_SIZE = 64
learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 64 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

X_train = []
Y_train = []
batch_array = []

def main():
    # Initialization of the board
    env = Board(4, 4)
    # Set the mines (use set_mines_about) about (2,2)

    # create the model using DQN_setup
    # and target_model
    # HINT: use board3D function for input dimension

    # dig at the 
    env.dig(2,2)

    # try using model.predict function to choose the next move (random at first)
    # and get the best action using argmax

    # get the reward using env.dig function

    # TRAIN PROCESS
    # append the data to the batch_array
    # 1. create q_table based on current state
    # 2. calculate the potential reward using target model
    # 3. update the q_table with reward + DISCOUNT FACTOR * potential reward
    # 4. append to the current state to the X_train and updated q_table to the Y_train

    model.fit(np.array(X_train), np.array(Y_train), batch_size=1)


if __name__ == "__main__":
    main()