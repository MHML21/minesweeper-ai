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

gamestate_to_reward = {Board.GAME_LOST: -1,
                       Board.INVALID_MOVE: -1,
                       Board.GAME_CONT: 1,
                       Board.GAME_WON: 2}

X_train = []
Y_train = []
batch_array = []

def main():
    env = Board(4, 4)
    env.set_mines_about(2,2,1)
    state_im = env.board3D()
    model = DQN_setup(learn_rate, state_im.shape, 16, CONV_UNITS, DENSE_UNITS)
    target_model = DQN_setup(learn_rate, state_im.shape, 16, CONV_UNITS, DENSE_UNITS)
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
    new_state, gamestate = env.dig(math.floor(action / 4), action % 4)
    new_state = env.board3D()
    
    reward = gamestate_to_reward[gamestate]
    
    # train
    batch_array.append((state_im, action, reward, new_state))
    current_states = np.array([transition[0] for transition in batch_array])
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([transition[3] for transition in batch_array])
    future_qs_list = target_model.predict(new_current_states)
    for i, (current_state, action, reward, new_current_states) in enumerate(batch_array):
            if reward % 2:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            X_train.append(current_state)
            Y_train.append(current_qs)

    model.fit(np.array(X_train), np.array(Y_train), batch_size=1)
    print("3")
    # print(env)


if __name__ == "__main__":
    main()
