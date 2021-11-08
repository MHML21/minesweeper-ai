from minesweeper_env import *
from DQNsetup import *
import numpy as np
import pandas as pd
from tensorboard import *

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
    env = Minesweeper(4, 4, 4)
    model = DQN_setup(learn_rate, env.state_im.shape, env.ntiles, CONV_UNITS, DENSE_UNITS)
    # print("init")
    # print(env)
    env.step(3)
    temp_state_im = env.state_im
    # get action
    board = temp_state_im.reshape(1, env.ntiles)
    moves = model.predict(np.reshape(temp_state_im, (1, env.nrows, env.ncols, 1)))
    print(type(moves))
    print("moves:", moves)
    moves[board!=-0.125] = np.min(moves) # set already clicked tiles to min value
    action = np.argmax(moves)
    print("action:", action)

    # main
    new_state, reward, done = env.step(action)

    # train
    current_qs_list = model.predict(np.reshape(temp_state_im, (1, env.nrows, env.ncols, 1)))
    current_qs = current_qs_list[0]
    print(type(current_qs))
    print("q_table:", current_qs)
    current_qs[action] = reward
    X_train.append(temp_state_im)
    Y_train.append(current_qs)
    model.fit(np.array(X_train), np.array(Y_train), batch_size=1,
                       shuffle=False, verbose=0, callbacks=[tensorboard]\
                       if done else None)
    print("3")
    # print(env)

if __name__ == "__main__":
    main()
