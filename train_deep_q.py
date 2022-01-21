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

board = Board(8,8)
board.set_mines_about(4,4,random.randint(10,40))

X_train = []
Y_train = []

total_reward = 0

# play 100000 moves to train the model
for i in range(100000):
    snapshot_row = 0
    snapshot_col = 0
    # pick a random window (snapshot) to consider
    snapshot = board.get_snapshot(snapshot_row, snapshot_col)


    # valid = bool of if there is any unopened tile
    valid = -1 in snapshot
    # where model actually predicts, rewards is in 1D array
    snapshot_temp = np.reshape(snapshot,(1,64))
    snapshot3D = np.reshape(snapshot_temp,(SNAPSHOT_SIZE,SNAPSHOT_SIZE,1))
    rewards = model.predict(np.reshape(snapshot3D,(1,SNAPSHOT_SIZE,SNAPSHOT_SIZE,1)))[0]

    if random.uniform(0,1) < 0.8:
        # find the best move
        action = argmax(rewards)
    else:
        # random move
        action = randint(0,SNAPSHOT_SIZE**2 - 1)
    
    
    row = math.floor(action/SNAPSHOT_SIZE)
    col = action % SNAPSHOT_SIZE
    
    gamestate = board.dig(row,col)
    reward = 0
    
    if gamestate == board.GAME_CONT:
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
    next_state = board.get_snapshot(snapshot_row, snapshot_col)
    next_state_rewards = model.predict(np.reshape(next_state,(1,SNAPSHOT_SIZE,SNAPSHOT_SIZE,1)))[0]
    rewards[action] = reward + 0.01*max(next_state_rewards)
    

    X_train.append(np.reshape(snapshot,(SNAPSHOT_SIZE,SNAPSHOT_SIZE,1)).astype(object))
    Y_train.append(rewards)
    
    if (i%4 == 0 and not len(X_train) == 0):
        model.fit(np.array(X_train), np.array(Y_train))
        X_train = []
        Y_train = []
        #print(total_reward)
        total_reward = 0
    if (i%100 == 0):
        print(rewards)
    
    # epsilon = max(EPSILON_MIN, epsilon*0.9998)
        
model.save("model_bigger_snapshot")  
    