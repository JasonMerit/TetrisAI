import numpy as np

height, width = 16, 10
current_piece = (4, 0, 0)


def new_board():
    board = np.zeros([height + 2, width])
    wall = np.ones([height + 2, 2])
    floor = np.ones([2, width + 5])
    board = np.c_[np.ones(height + 2), wall, board, wall]
    board = np.vstack((board, floor))
    return board


class TET:
    def search_step(self, action):
        if action == 1:
            self.piece.x -= 1  # Move left
        elif action == 2:
            self.piece.x += 1  # Move right
        elif action == 3:
            self.piece.y += 1  # Move down
        elif action == 4:
            self.piece.rotate()  # Rotate clockwise
        elif action == 5:
            self.piece.rotate(False)  # Rotate counter-clockwise

    def BFS(self):
        """
        Breadth first search, using two lists for the current branches one to loop
        through and one to append to, one list for all the valid states visited
        to prevent a case of two steps forward and two steps back infinite looping.
        If a valid position where the piece is placed is found, its coordinates
        and rotation are appended to the actions list, and its features are appended
        to the Features list.

        returns actions and Features lists separately
        """
        current_pos = self.piece.get_pos(0)
        append_list = [current_pos]
        loop_list = []
        visited = [current_pos]
        actions = []
        Features = []

        while len(append_list) > 0:
            loop_list = append_list
            for state in loop_list:



        return actions, Features

        # Start:
        # add current position to loop_list, with action = 0
        # Loop through 1-5:
        # take corresponding action so long as it is not the previous action
        # check each step for validity, if valid:
        #   append (x, y, rotation, previous step) to loop_list
        # check is piece is placed:
        #   append to actions list
        #   remove from loop_list
        # take another step, this time only using the four other operations
        # end when no step is valid
        #
        #
