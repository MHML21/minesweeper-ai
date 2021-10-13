#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:17:17 2021

@author: alanlee
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import argmax
import math
from minesweeper import Board

model = keras.models.load_model("model1")

b = Board(16,16)
b.set_mines_about(4,4,30)
b.d(4,4)
np.set_printoptions(precision=5)

def predict(board, row,col):
    return model.predict(np.array([board.get_snapshot(row, col).flatten()]))[0]

def play(board,snap_row,snap_col):
    action = argmax(predict(board,snap_row,snap_col))
    row = math.floor(action/4)
    col = action % 4
    if board.dig_at_snapshot(snap_row,snap_col,row,col) == board.INVALID_MOVE:
        print("invalid")
    print(row)
    print(col)
    print(board)
