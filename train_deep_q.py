#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:35:32 2021

@author: alanlee
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import argmax
from minesweeper import Board
from random import randint
import random
import math

SNAPSHOT_SIZE = 8


init = tf.keras.initializers.HeUniform()
model = keras.Sequential([
     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape = (8,8,1)),
     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
     keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
     keras.layers.Flatten(),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dense(SNAPSHOT_SIZE**2, activation='linear')])

model.build((None,SNAPSHOT_SIZE**2))
model.compile(optimizer = "adam", loss = "huber")


#model  = keras.models.load_model("model1")

board = Board(16,16)
board.set_mines_about(4,4,random.randint(10,40))

X_train = []
Y_train = []

total_reward = 0

# play 100000 moves to train the model
for i in range(100000):
    snapshot_row = randint(0,board.rows - SNAPSHOT_SIZE -1)
    snapshot_col = randint(0,board.cols - SNAPSHOT_SIZE - 1)
    # pick a random window (snapshot) to consider
    snapshot = board.get_snapshot(snapshot_row, snapshot_col)

    # checks if the snapshot is unopened, or unknown
    is_guess = np.array_equiv(snapshot, np.full((SNAPSHOT_SIZE,SNAPSHOT_SIZE), -1,dtype = float))
    # valid = bool of if there is any unopened tile
    valid = -1 in snapshot
    # where model actually predicts, rewards is in 1D array
    rewards = model.predict(np.reshape(snapshot,(1,8,8,1)))[0]
    print(rewards)
    '''
    if random.uniform(0,1) < 0.8:
        # find the best move
        action = argmax(rewards)
    else:
        # random move
        action = randint(0,SNAPSHOT_SIZE**2 - 1)
    
    
    row = math.floor(action/SNAPSHOT_SIZE)
    col = action % SNAPSHOT_SIZE
    # converting local row, col to global row, col
    # and call function dig()
    gamestate = board.dig_at_snapshot(snapshot_row,snapshot_col,row,col)
    reward = 0
    
    if gamestate == board.GAME_CONT:
        if is_guess:
            reward = 0
        else:
            reward = 1
    elif gamestate == board.INVALID_MOVE:
        reward = -1
    elif gamestate == board.GAME_LOST:
        reward = -1
        board = Board(16,16)
        board.set_mines_about(4,4,random.randint(10,40))
    elif gamestate == board.GAME_WON:
        reward = 1
        board = Board(16,16)
        board.set_mines_about(4,4,random.randint(10,40))
    
    # don't want to do this if this is not valid
    total_reward += reward
    rewards[action] = reward
    
    if valid and not is_guess :
        X_train.append(snapshot.flatten())
        Y_train.append(rewards)
    # reduce the number of moves that are is_guess or invalid
    elif random.uniform(0,1) > 0.95:
        X_train.append(snapshot.flatten())
        Y_train.append(rewards)
    
    if (i%4 == 0 and not len(X_train) == 0):
        model.fit(np.array(X_train), np.array(Y_train))
        X_train = []
        Y_train = []
        #print(total_reward)
        total_reward = 0
    if (i%100 == 0):
        print(rewards)
    '''
        
model.save("model_bigger_snapshot")  
    