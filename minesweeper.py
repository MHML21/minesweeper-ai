# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random
import math
from pandas import *

class Board():
    
    GAME_LOST = 0
    GAME_CONT = 1
    GAME_WON = 2
    INVALID_MOVE = 3
    
    SNAPSHOT_SIZE = 8
    
    def __init__(self,rows,cols):
        self.rewards={'WIN':1, 'LOSE':-1, 'PROGRESS':0.3, 'GUESS':-0.3, 'NO_PROGRESS' : -0.3}
        self.rows = rows
        self.cols = cols
        self.ntiles = rows*cols
        self.board = np.full((rows,cols), -1,dtype = float) # what user sees
        # -1 meaning unopened, 0-8 indicates numbers
        self.mines = np.full((rows,cols), 0) # 0 means not bomb, 1 means bomb
        self.opened = 0
        self.n_wins = 0
        
        for row in range(rows):
            for col in range(cols):
                self.mines[(row,col)] = 0

    def reset(self):
        self.board = np.full((self.rows,self.cols), -1,dtype = float) # what user sees
        # -1 meaning unopened, 0-8 indicates numbers
        self.mines = np.full((self.rows,self.cols), 0) # 0 means not bomb, 1 means bomb
        self.opened = 0
        
        for row in range(self.rows):
            for col in range(self.cols):
                self.mines[(row,col)] = 0

    def set_mines_about(self,row_center,col_center,num_mines):
        self.num_mines = num_mines
        sample_points = list(range((self.rows)*(self.cols)))
        for i in range(-1,2):
            for j in range(-1,2):
                if self.is_in_bounds(row_center+i,col_center+j):
                    sample_points.remove((row_center+i)*self.cols + (col_center+j))

        # Set the mines on the board that is not around the center point
        cords = random.sample(sample_points, num_mines)
        for cord in cords:
            row = math.floor(cord/self.cols)
            col = cord % self.cols
            self.mines[(row,col)] = 1

        # open the center point
        self.dig(row_center, col_center)
        
    def dig(self,row,col):
        print("Opening ( %d, %d ) ..." % (row, col))

        # If the tile is a bomb
        if self.mines[row,col] == 1:
            print("GAME LOST, BOMB DETECTED")
            board3d = self.board3D()
            return board3d, self.rewards['LOSE'], True
        
        elif self.board[row,col] == -1: # NOT A BOMB && UNOPENED
            counter = 0
            self.opened += 1
            for i in range(-1,2):
                for j in range(-1,2):
                    if self.is_in_bounds(row+i,col+j):
                        if self.mines[(row + i,col+j)] == 1:
                            counter += 1
            self.board[(row,col)] = counter/8
            if self.opened == (self.ntiles - self.num_mines):
                self.n_wins += 1
                print("GAME WON, CONGRATULATIONS!!")
                board3d = self.board3D()
                return board3d, self.rewards['WIN'], True
            
            # If there are no bombs around it, open until they find one
            if counter == 0:
                for i in range(-1,2):
                    for j in range(-1,2):
                        if self.is_in_bounds(row+i,col+j):
                            if self.board[(row+i,col+j)] == -1:
                                self.dig(row + i,col+j)
            print("Progressed ...")
            board3d = self.board3D()
            return board3d, self.rewards['PROGRESS'], False
        else:
            print("INVALID MOVE")
            board3d = self.board3D()
            return board3d, self.rewards['NO_PROGRESS'], False

    def board3D(self):
        return np.reshape(self.board,(self.rows,self.cols,1))

    def printBoard(self):
        print(self.board * 8)

    def printMines(self):
        print(self.mines)

    def d(self,row,col):
        self.dig(row,col)
        print(self)
        
    def is_in_bounds(self,row,col):
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return True
    
    def get_board3d(self):
        return np.reshape(self.board,(self.rows,self.cols,1))
    def get_snapshot(self,row,col):
        assert(row >= 0 and row <= self.rows + self.SNAPSHOT_SIZE)
        assert(col >= 0 and col <= self.cols + self.SNAPSHOT_SIZE)
        return self.board[row:row+self.SNAPSHOT_SIZE, col:col+self.SNAPSHOT_SIZE]
    
    def dig_at_snapshot(self,snap_row,snap_col,row_in_snap,col_in_snap):
        return self.dig(snap_row + row_in_snap, snap_col + col_in_snap)
      
    def __str__(self):
        print(self.board * 8)
        return""
'''      
np.set_printoptions(precision=3)
b =  Board(16,16)
b.set_mines_about(4,4,40)
b.d(4,4)'''


