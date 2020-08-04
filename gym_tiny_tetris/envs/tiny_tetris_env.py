try:
    from importlib.resources import open_text
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    from importlib_resources import open_text
import random
import sys

import gym
from gym import error, spaces, utils
import numpy as np

from . import inp


class TinyTetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, use_file=1):
        """Initializes the environment and the random seed."""
        self.piece_ptr = 0
        self.piece_list = list(
            map(int, open_text(inp, f'tiny.i{use_file}').readlines()[1:]))

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(10, 9),
            dtype=int)
        self.reset()

    def reset(self):
        """Clears the board and sets the random seed."""
        self.score = 0
        self.board = [[0 for i in range(9)] for j in range(9)]

        # N.B. The piece types are 0-indexed
        self.piece_ptr = 0
        self.next_piece = self._get_next_piece()
        return self._get_state()

    def render(self, mode='human', close=False):
        """Just prints the current game board."""
        print(f'Score: {self.score}')
        for row in self.board:
            print(''.join(map(str, row)))
        print()

    def step(self, action):
        """
        This method steps the game forward one step and
        shoots a bubble at the given angle.
        Parameters
        ----------
        action : int
            The action is an integer between 0 and 8 (inclusive), that
            decides where the current piece should drop.
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing the
                state of the environment.
            reward (float) :
                amount of reward achieved by the previous action.
            episode_over (bool) :
                whether it's time to reset the environment again.
            info (dict) :
                diagnostic information useful for debugging.
        """
        # Test whether action is valid
        if action < 0 or action > 8:
            raise Exception(f'Invalid action: {action}')

        # End the game if we can't place the piece
        if not self._can_place(self.next_piece, action):
            return self._get_state(), -10, True, {}

        # Place the piece
        reward = self._place_piece(self.next_piece, action)

        return self._get_state(), reward, reward > 1e5, {}

    def _can_place(self, type, column):
        """Given the piece type and column, determines whether it's valid."""
        if type == 0:
            return not (self.board[0][column])
        elif type == 1:
            return not (self.board[0][column] or self.board[1][column])
        elif type == 2:
            return not (column > 7 or self.board[0][column] or self.board[0][column + 1])
        elif type == 3:
            return not (self.board[0][column] or self.board[1][column] or self.board[2][column])
        elif type == 4:
            return not (column > 6 or self.board[0][column] or self.board[0][column + 1] or self.board[0][column + 2])
        elif type == 5:
            return not (column > 7 or self.board[0][column] or self.board[1][column] or self.board[0][column + 1] or self.board[1][column + 1])
        elif type == 6:
            return not (column > 7 or self.board[0][column] or self.board[1][column] or self.board[0][column + 1] or self.board[1][column + 1])
        elif type == 7:
            return not (column > 7 or self.board[0][column] or self.board[1][column] or self.board[0][column + 1])
        elif type == 8:
            return not (column > 7 or self.board[0][column] or self.board[1][column + 1] or self.board[0][column + 1])

    def _place_piece(self, type, column):
        """Places the piece in the given column."""
        reward = 1

        if type == 0:
            for i in range(9):
                if i == 8 or self.board[i + 1][column]:
                    self.board[i][column] = 1
                    break
        elif type == 1:
            for i in range(1, 9):
                if i == 8 or self.board[i + 1][column]:
                    self.board[i][column] = self.board[i - 1][column] = 1
                    break
        elif type == 2:
            for i in range(9):
                if i == 8 or self.board[i + 1][column] or self.board[i + 1][column + 1]:
                    self.board[i][column] = self.board[i][column + 1] = 1
                    break
        elif type == 3:
            for i in range(2, 9):
                if i == 8 or self.board[i + 1][column]:
                    self.board[i][column] = self.board[i -
                                                       1][column] = self.board[i - 2][column] = 1
                    break
        elif type == 4:
            for i in range(9):
                if i == 8 or self.board[i + 1][column] or self.board[i + 1][column + 1] or self.board[i + 1][column + 2]:
                    self.board[i][column] = self.board[i][column +
                                                          1] = self.board[i][column + 2] = 1
                    break
        elif type == 5:
            for i in range(1, 9):
                if i == 8 or self.board[i + 1][column] or self.board[i + 1][column + 1]:
                    self.board[i][column] = self.board[i][column +
                                                          1] = self.board[i - 1][column] = 1
                    break
        elif type == 6:
            for i in range(1, 9):
                if i == 8 or self.board[i + 1][column] or self.board[i + 1][column + 1]:
                    self.board[i][column] = self.board[i][column +
                                                          1] = self.board[i - 1][column + 1] = 1
                    break
        elif type == 7:
            for i in range(1, 9):
                if i == 8 or self.board[i + 1][column] or self.board[i][column + 1]:
                    self.board[i][column] = self.board[i -
                                                       1][column] = self.board[i - 1][column + 1] = 1
                    break
        elif type == 8:
            for i in range(1, 9):
                if i == 8 or self.board[i][column] or self.board[i + 1][column + 1]:
                    self.board[i - 1][column] = self.board[i][column +
                                                              1] = self.board[i - 1][column + 1] = 1
                    break

        # Erase cleared lines
        self.board = list(filter(lambda row: 0 in row, self.board))
        while len(self.board) != 9:
            self.board.insert(0, [0 for i in range(9)])
            reward += 50

        # Next piece
        self.next_piece = self._get_next_piece()
        if self.piece_ptr == len(self.piece_list):
            reward += 1e5
        self.score += 1

        return reward

    def _get_state(self):
        """
        First 9 rows: The board
        10th row: One-hot vector indicating next piece
        """
        piece_arr = [0 for i in range(9)]
        piece_arr[self.next_piece] = 1
        return np.vstack([self.board, piece_arr])

    def _get_next_piece(self):
        """Gets the next piece."""
        self.piece_ptr += 1
        if self.piece_ptr == len(self.piece_list):
            self.piece_ptr = 0
        return self.piece_list[self.piece_ptr - 1] - 1
