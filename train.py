# RUN THE FILES IF THEY HAVEN'T
from tensorflow.keras.models import load_model
from DQNAgent import *
import math
from tqdm import tqdm
import pickle

import warnings
warnings.filterwarnings('ignore')

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 10_000 # save model and replay every 10,000 episodes

def main():
    # INITIALIZING EVERTHING CELL

    # Initialize the Board (9x9 matrix)
    env = Board(9, 9)

    # Set the mines about random coordinate
    # Assume user clicked the random coordinate as the first tile
    f_row, f_col = np.random.randint(env.rows, size=2)
    print("First row: %d, First Col: %d" % (f_row, f_col))
    env.set_mines_about(f_row, f_col,10) # set_mines_about(self,row_center,col_center,num_mines)
    # print("Mines: ")
    # env.printMines()
    # print("Board: ")
    # env.printBoard()
    state_im = env.board3D() # board is currently 2D, making it 3D by (row, col, 1)

    agent = DQNAgent(env)

    progress_list, wins_list, ep_rewards = [], [], []
    n_clicks = 0

    # PLAY THE GAME!!! (# episodes Games)
    for episode in tqdm(range(1,episodes+1), unit='episode'):
        agent.tensorboard.step = episode
        
        env.reset()
        f_row, f_col = np.random.randint(env.rows, size=2)
        env.set_mines_about(f_row, f_col,10)
        done = False
        ep_reward = 0
        
        past_n_wins = env.n_wins

        # play until lose
        while not done:
            
            current_state = env.board3D()
            
            action = agent.get_action(current_state)

            # Retrieve the next step and reward
            new_state, reward, done = env.dig(math.floor(action / env.cols), action % env.cols)
            # print("\nREWARD: ", reward)
            ep_reward += reward

            # append the data to batch_array
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            n_clicks += 1
            
        progress_list.append(env.n_progress) # n of non-guess moves
        ep_rewards.append(ep_reward)
        
        # print("Number of Wins :", env.n_wins)
        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(agent.replay_memory) < MEM_SIZE_MIN:
            # print("SKIP after Training Process")
            continue

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            agent.tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = agent.learn_rate,
                epsilon = agent.epsilon)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

        if not episode % SAVE_MODEL_EVERY:
            with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                pickle.dump(agent.replay_memory, output)

            agent.model.save(f'models/{MODEL_NAME}.h5')
        
    print("Number of Wins :", env.n_wins)



if __name__ == "__main__":
    main()
