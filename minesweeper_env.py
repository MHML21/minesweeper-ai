import random
import numpy as np
import pandas as pd

# TO DO : line # 48, 61, 76, 139, 187
class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
        # based on https://github.com/jakejhansen/minesweeper_solver
        rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines # number of mines
        self.grid = self.init_grid() # only keeps track of mines
        self.board = self.get_board() # entire board under the tiles 
        # STATE_IM  = matrix with value ranging from -0.25 to 1 ((-2 to 8) / 8 )
        # -2 indicates bomb, -1 indicates unknown, 0-8 indicates number of bombs adject to the tile
        # Since we divide the entire matrix by 8, STATE_IM would have value of:
        # -0.25 indicates bomb, -0.125 indicates unknown, 0-1 indicates number of bombs adject to the tile
        # STATE = 1d array of objects with coord and value ("U" or "B")
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0 # number of clicks
        self.n_progress = 0
        self.n_wins = 0

        self.rewards = rewards

    def init_grid(self):
        # fill up grid with all zeros
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        # randomly place the mines on the board
        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            # make sure you are not placing two mines on the same spot
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        # Think of coord as pair (x,y)
        x,y = coord[0], coord[1]

        neighbors = []
        # Get all the tiles that are adjacent to the coordinate (x,y)
        # WRITE YOUR CODE HERE

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        # Set up the board so that all tiles (except tiles with a bomb) 
        # have an integer (from 0 to 8) that indicates the number of bombs that are adjacent to the tile
        # WRITE YOUR CODE HERE

        return board

    def get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''

        state_im = [t['value'] for t in state]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        # Set all the values of Unknown tiles to -1
        # Set all the values of Bomb tiles to -2
        # WRITE YOUR CODE HERE

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        return state_im

    def init_state(self):
        unsolved_array = np.full((self.nrows, self.ncols), 'U', dtype='object')

        state = []
        for (x, y), value in np.ndenumerate(unsolved_array):
            state.append({'coord': (x, y), 'value':value})

        state_im = self.get_state_im(state)

        return state, state_im

    def color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self, state_im):
        state = state_im * 8.0
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self.color_state))

    def click(self, action_index):
        coord = self.state[action_index]['coord']
        value = self.board[coord]

        # ensure first move is not a bomb
        if (value == 'B') and (self.n_clicks == 0):
            grid = self.grid.reshape(1, self.ntiles)
            move = np.random.choice(np.nonzero(grid!='B')[1])
            coord = self.state[move]['coord']
            value = self.board[coord]
            self.state[move]['value'] = value
        # else: (UNCOMMENT THIS)
            # make state equal to board at given coordinates
            # WRITE YOUR CODE HERE

        # reveal all neighbors if value is 0
        if value == 0.0:
            self.reveal_neighbors(coord, clicked_tiles=[])

        self.n_clicks += 1

    def reveal_neighbors(self, coord, clicked_tiles):
        processed = clicked_tiles
        state_df = pd.DataFrame(self.state)
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows) and
                    ((row, col) not in processed)):

                    # prevent redundancy for adjacent zeros
                    processed.append((row,col))

                    index = state_df.index[state_df['coord'] == (row,col)].tolist()[0]

                    self.state[index]['value'] = self.board[row, col]

                    # recursion in case neighbors are also 0
                    if self.board[row, col] == 0.0:
                        self.reveal_neighbors((row, col), clicked_tiles=processed)

    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()

    def step(self, action_index):
        done = False
        coords = self.state[action_index]['coord']

        current_state = self.state_im

        # get neighbors before action
        neighbors = self.get_neighbors(coords)

        # WRITE YOUR CODE HERE (3 lines of code)
        # 1. click the given coordinate (action index) and reveal its value
        # HINT: use click() function

        # 2. get the new state image and save the return value to "new_state_im"

        # 3. update the self.state_im to new_state_im

        if self.state[action_index]['value']=='B': # if lose
            reward = self.rewards['lose']
            done = True

        elif np.sum(new_state_im==-0.125) == self.n_mines: # if win
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(self.state_im == -0.125) == np.sum(current_state == -0.125):
            reward = self.rewards['no_progress']

        else: # if progress
            if all(t==-0.125 for t in neighbors): # if guess (all neighbors are unsolved)
                reward = self.rewards['guess']

            else:
                reward = self.rewards['progress']
                self.n_progress += 1 # track n of non-isoloated clicks

        return self.state_im, reward, done
