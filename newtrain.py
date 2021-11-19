#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:10:18 2021

@author: alanlee
"""

from minesweeper import Board
# from minesweeper_env import *
from DQNsetup import *
import numpy as np
import pandas as pd
import math
import random

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

def main():
    env = Board(4, 4)
    env.set_mines_about(2,2,1)
    state_im = env.board3D()
    model = DQN_setup(learn_rate, state_im.shape, 16, CONV_UNITS, DENSE_UNITS)
    # print("init")
    # print(env)
    env.dig(2,2)
    temp_state_im = state_im
    # get action
    board = temp_state_im.reshape(1, 16)
    moves = model.predict(np.reshape(temp_state_im, (1, env.rows, env.cols, 1)))
    print(type(moves))
    print("moves:", moves)
    # moves[board!=-1] = np.min(moves) # set already clicked tiles to min value
    action = np.argmax(moves)
    print("action:", action)

    # main
    gamestate = env.dig(math.floor(action / 4), action % 4)

    # train
    current_qs_list = model.predict(np.reshape(temp_state_im, (1, env.rows, env.cols, 1)))
    current_qs = current_qs_list[0]
    print("list_q_table:", current_qs_list)
    print(type(current_qs))
    print("q_table:", current_qs)
    
    reward = 0
    
    if gamestate == env.GAME_CONT:
        reward = 1
    elif gamestate == env.INVALID_MOVE:
        reward = -1
    elif gamestate == env.GAME_LOST:
        reward = -1
        env = Board(4,4)
        env.set_mines_about(2,2,random.randint(0,4))
    elif gamestate == env.GAME_WON:
        reward = 1
        env = Board(4,4)
        env.set_mines_about(2,2,random.randint(0,4))
    
    current_qs[action] = reward
    X_train.append(temp_state_im)
    Y_train.append(current_qs)
    model.fit(np.array(X_train), np.array(Y_train), batch_size=1)
    print(Y_train)
    #                   shuffle=False, verbose=0)
    print("3")
    # print(env)


if __name__ == "__main__":
    main()
