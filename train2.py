# RUN THE FILES IF THEY HAVEN'T
import random
import numpy as np
import pandas as pd
import math
from minesweeper import Board
from DQNsetup import *
from model_tensorboard import *
from collections import deque
from tqdm import tqdm
import pickle

import warnings
warnings.filterwarnings('ignore')

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer
# episodes = 100_000

# Learning settings
BATCH_SIZE = 64
# learn_rate = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
# epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 128 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

# Default model name
MODEL_NAME = f'conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}'

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

def main():
    # INITIALIZING EVERTHING CELL

    # Initialize the Board (4x4 matrix)
    env = Board(9, 9)

    learn_rate = 0.01
    epsilon = 0.95
    episodes = 100_000

    # Set the mines about (2,2)
    # Assume user clicked coordinate (2,2) as the first tile
    f_row, f_col = np.random.randint(env.rows, size=2)
    print("First row: %d, First Col: %d" % (f_row, f_col))
    env.set_mines_about(f_row, f_col,10) # set_mines_about(self,row_center,col_center,num_mines)
    print("Mines: ")
    env.printMines()
    print("Board: ")
    env.printBoard()
    state_im = env.board3D() # board is currently 2D, making it 3D by (row, col, 1)

    # Initialize the model 
    model = DQN_setup(learn_rate, state_im.shape, env.ntiles, CONV_UNITS, DENSE_UNITS)

    # Initialize the model that would always be ahead (looking at the future)
    target_model = DQN_setup(learn_rate, state_im.shape, env.ntiles, CONV_UNITS, DENSE_UNITS)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=MEM_SIZE)
    target_update_counter = 0

    tensorboard = ModifiedTensorBoard(
                log_dir=f'logs\\{MODEL_NAME}', profile_batch=0)

    progress_list, wins_list, ep_rewards = [], [], []

    # PLAY THE GAME!!! (# episodes Games)
    for episode in tqdm(range(1,episodes+1), unit='episode'):
        tensorboard.step = episode
        
        env.reset()
        f_row, f_col = np.random.randint(env.rows, size=2)
        env.set_mines_about(f_row, f_col,15)
        done = False
        ep_reward = 0
        
        past_n_wins = env.n_wins

        # play until lose
        while not done:
            
            current_state = env.board3D()
            
            # get action
            board = env.board3D().reshape(1, env.ntiles)
            # print("Board: ", board)

            # Select the unopened tiles
            unopened_tiles = [i for i, value in enumerate(board[0]) if value==-1]
            # print("Unopened Tiles: ", unopened_tiles)

            rand = np.random.random() # number from 0 to 1

            if rand < epsilon:
                # print("\nUsed Random")
                move = np.random.choice(unopened_tiles)
            else:
                # print("\nUsed Model To Predict")
                moves = model.predict(np.reshape(current_state, (1, env.rows, env.cols, 1)))
                # print(type(moves))
                # print("moves:", moves)
                # Disregard all the opened tiles
                moves[board!=-1] = np.min(moves)
                # Pick a tile with the best probability
                move = np.argmax(moves)


            # print("\naction: ", move)
            # print("Board: ", env.board)
            # print("Mines: ", env.mines)

            # Retrieve the next step and reward
            new_state, reward, done = env.dig(math.floor(move / env.cols), move % env.cols)
            # print("\nREWARD: ", reward)
            ep_reward += reward

            # append the data to batch_array
            replay_memory.append((current_state, move, reward, new_state, done))

            # Train
            if len(replay_memory) < MEM_SIZE_MIN:
                # print("SKIP in Training Process")
                continue

            # Get the random sample of batches
            batch = random.sample(replay_memory, BATCH_SIZE)

            # Q table for current model using the batches
            current_states = np.array([transition[0] for transition in batch])
            current_qs_list = model.predict(current_states)
            # print(current_qs_list.shape)
            # print("\nCurrent Q Table: ", current_qs_list)

            # Q table for future model
            new_current_states = np.array([transition[3] for transition in batch])
            future_qs_list = target_model.predict(new_current_states)
            # print(future_qs_list.shape)
            # print("\nFuture Q Table: ", future_qs_list)
            
            X_train = [] # Feature
            Y_train = [] # Label

            for i, (current_state, action, reward, new_current_states, done) in enumerate(batch):
                if not done:
                    max_future_q = np.max(future_qs_list[i])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                current_qs = current_qs_list[i]
                current_qs[action] = new_q

                X_train.append(current_state)
                Y_train.append(current_qs)

            model.fit(np.array(X_train), np.array(Y_train), batch_size=BATCH_SIZE,
                      shuffle=False, verbose=0, callbacks=[tensorboard]\
                      if done else None)

            # updating to determine if we want to update target_model yet
            if done:
                target_update_counter += 1

            if target_update_counter > UPDATE_TARGET_EVERY:
                target_model.set_weights(model.get_weights())
                target_update_counter = 0

            # decay learn_rate
            learn_rate = max(LEARN_MIN, learn_rate*LEARN_DECAY)

            # decay epsilon
            epsilon = max(EPSILON_MIN, epsilon*EPSILON_DECAY)
            
        progress_list.append(env.n_progress) # n of non-guess moves
        ep_rewards.append(ep_reward)
        
        # print("Number of Wins :", env.n_wins)
        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(replay_memory) < MEM_SIZE_MIN:
            # print("SKIP after Training Process")
            continue

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = learn_rate,
                epsilon = epsilon)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

        if not episode % SAVE_MODEL_EVERY:
            with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(replay_memory, output)

            model.save(f'models/{MODEL_NAME}.h5')
        
    print("Number of Wins :", env.n_wins)



if __name__ == "__main__":
    main()
