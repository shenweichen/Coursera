"""
Clone of 2048 game.
"""

import poc_2048_gui
import random


# Directions, DO NOT MODIFY
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Offsets for computing tile indices in each direction.
# DO NOT MODIFY this dictionary.
OFFSETS = {UP: (1, 0),
           DOWN: (-1, 0),
           LEFT: (0, 1),
           RIGHT: (0, -1)}

def merge(line):
    """
    Function that merges a single row or column in 2048.
    """
    # replace with your code
    ans = [0]*len(line)
    pre = 0
    for cur in range(0, len(line)):
        if line[cur] == 0:
            continue
        if ans[pre] == 0:
            ans[pre] = line[cur]
        elif line[cur] == ans[pre]:
            ans[pre] += line[cur]
            pre += 1
        else:
            pre += 1
            ans[pre] = line[cur]
    return ans

class TwentyFortyEight:
    """
    Class to run the game logic.
    """

    def __init__(self, grid_height, grid_width):
        # replace with your code
        self._height = grid_height
        self._width = grid_width
        self._initial_tiles = {UP:[(0,i) for i in range(0,self._width)],
        DOWN:[(self._height-1,i) for i in range(self._width)],
        LEFT:[(i,0) for i in range(self._height)],
        RIGHT:[(i,self._width-1) for i in range(self._height)]}
        self.reset()

    def reset(self):
        """
        Reset the game so the grid is empty except for two
        initial tiles.
        """
        # replace with your code
        self._cells = [[0 for dummy_col in range(self._width)] for dummy_row in range(self._height)]
        self.new_tile()
        self.new_tile()

    def __str__(self):
        """
        Return a string representation of the grid for debugging.
        """
        # replace with your code
        stt = "["
        for dummy_row in range(self._height):
            stt+='['+','.join(map(str, self._cells[dummy_row]))+"]"
            if dummy_row < self._height - 1:
                stt+="\n "
        return stt+']'

    def get_grid_height(self):
        """
        Get the height of the board.
        """
        # replace with your code
        return self._height

    def get_grid_width(self):
        """
        Get the width of the board.
        """
        # replace with your code
        return self._width

    def move(self, direction):
        """
        Move all tiles in the given direction and add
        a new tile if any tiles moved.
        """
        # replace with your code
        if direction == UP or direction == DOWN :
            range_n = self._height
        else:
            range_n = self._width
        changed = False
        for tile in self._initial_tiles[direction]:
            cur_row,cur_col = tile
            tile_value = []
            for dummy_i in range(0,range_n):
                tile_value.append(self._cells[cur_row][cur_col])
                cur_row += OFFSETS[direction][0]
                cur_col += OFFSETS[direction][1]
            orgin_value = [value for value in tile_value]
            tile_value = merge(tile_value)
            cur_row,cur_col = tile
            for dummy_i in range(0,range_n):
                self._cells[cur_row][cur_col] = tile_value[dummy_i]
                cur_row += OFFSETS[direction][0]
                cur_col += OFFSETS[direction][1]
            if orgin_value != tile_value:
                changed = True
        if changed:
            self.new_tile()

    def new_tile(self):
        """
        Create a new tile in a randomly selected empty
        square.  The tile should be 2 90% of the time and
        4 10% of the time.
        """
        # replace with your code
        random.seed()
        while True:
            row,col = random.randint(0,self._height-1),random.randint(0,self._width-1)
            if self._cells[row][col] == 0:
                ans = random.randint(0, 9)
                self._cells[row][col] = 4 if ans == 0 else 2
                break

    def set_tile(self, row, col, value):
        """
        Set the tile at position row, col to have the given value.
        """
        # replace with your code
        self._cells[row][col] = value

    def get_tile(self, row, col):
        """
        Return the value of the tile at position row, col.
        """
        # replace with your code
        return self._cells[row][col]


poc_2048_gui.run_gui(TwentyFortyEight(4, 4))
