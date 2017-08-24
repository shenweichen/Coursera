# Do not import any modules. If you do, the tester may reject your submission.

# Constants for the contents of the maze.

# The visual representation of a wall.
WALL = '#'

# The visual representation of a hallway.
HALL = '.'

# The visual representation of a brussels sprout.
SPROUT = '@'

# Constants for the directions. Use these to make Rats move.

# The left direction.
LEFT = -1

# The right direction.
RIGHT = 1

# No change in direction.
NO_CHANGE = 0

# The up direction.
UP = -1

# The down direction.
DOWN = 1

# The letters for rat_1 and rat_2 in the maze.
RAT_1_CHAR = 'J'
RAT_2_CHAR = 'P'


class Rat:
    """ A rat caught in a maze. """

    # Write your Rat methods here.
    def __init__(self,symbol,row,col):
        self.symbol = symbol
        self.row = row
        self.col = col
        self.num_sprouts_eaten = 0
    def set_location(self,row,col):
        self.row = row
        self.col = col
    def eat_sprout(self,):
        self.num_sprouts_eaten += 1
    def __str__(self,):
        s = "{0} at ({1}, {2}) ate {3} sprouts.".format(self.symbol,self.row,self.col,self.num_sprouts_eaten)
        return s
        
class Maze:
    """ A 2D maze. """

    # Write your Maze methods here.
    def __init__(self,maze,rat_1,rat_2):
        self.maze = maze
        self.rat_1 = rat_1
        self.rat_2 = rat_2
        self.num_sprouts_left = 0
        for row in self.maze:
            self.num_sprouts_left += row.count(SPROUT)

    def is_wall(self,row,col):
        return self.maze[row][col] == WALL

    def get_character(self,row,col):
        if self.rat_1.row == row and self.rat_1.col == col:
            return self.rat_1.symbol
        if self.rat_2.row == row and self.rat_2.col == col:
            return self.rat_2.symbol
        return self.maze[row][col]

    def move(self,rat,vet,hor):
        if self.is_wall(rat.row+vet,rat.col+hor):
            return False
        if self.get_character(rat.row+vet,rat.col+hor) == SPROUT:
            rat.eat_sprout()
            self.num_sprouts_left -= 1
            self.maze[rat.row+vet][rat.col+hor] = HALL
        rat.set_location(rat.row+vet,rat.col+hor)
        return True
    def __str__(self,):
        maze = "\n".join(["".join(row)  for row in self.maze])
        maze += "\n"
        maze += str(self.rat_1)
        maze += "\n"
        maze += str(self.rat_2)
        return maze
