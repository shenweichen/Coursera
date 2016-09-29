"""
Monte Carlo Tic-Tac-Toe Player
"""

import random
import poc_ttt_gui
import poc_ttt_provided as provided

# Constants for Monte Carlo simulator
# You may change the values of these constants as desired, but
#  do not change their names.
NTRIALS = 1000         # Number of trials to run
SCORE_CURRENT = 2.0 # Score for squares played by the current player
SCORE_OTHER = 1.0   # Score for squares played by the other player
    
# Add your functions here.
def mc_trial(board,player):
    """
    mc_trial
    """
    empty_squares = board.get_empty_squares()
    while True:
        row,col = random.sample(empty_squares,1)[0]
        board.move(row,col,player)
        if board.check_win() != None:
            break
        player = provided.switch_player(player)
        empty_squares.remove((row,col))

def mc_update_scores(scores,board,player):
    """
    mc_update_scores
    """
    winner = board.check_win()
    if winner == provided.DRAW:
        return
    height=width = board.get_dim()
    for row in range(height):
        for col in range(width):
            state = board.square(row,col)
            if state == provided.EMPTY:
                continue
            if winner == player:
                if state == player:
                    scores[row][col] += SCORE_CURRENT
                else:
                    scores[row][col] -= SCORE_OTHER
            else:
                if state == player:
                    scores[row][col] -= SCORE_CURRENT
                else:
                    scores[row][col] += SCORE_OTHER


def get_best_move(board,scores):
    """
    get_best_move
    """
    empty_squares = board.get_empty_squares()
    max_board = [empty_squares[0]]
    max_score = scores[empty_squares[0][0]][empty_squares[0][1]]
    for item in empty_squares:
        row,col = item
        if scores[row][col] == max_score:
            max_board.append(item)
        elif scores[row][col] > max_score:
            max_board = [item]
            max_score = scores[row][col]
    return random.sample(max_board,1)[0]

def mc_move(board,player,trials):
    """
    mc_move
    """
    height=width = board.get_dim()
    scores = [[ 0 for dummy in range(width)] for dummy in range(height) ]
    for dummy in range(trials):
        test_board = board.clone()
        test_player = player
        mc_trial(test_board,test_player)
        mc_update_scores(scores,test_board,test_player)
    return get_best_move(board,scores)
# Test game with the console or the GUI.  Uncomment whichever 
# you prefer.  Both should be commented out when you submit 
# for testing to save time.
# provided.play_game(mc_move, NTRIALS, False)        
# poc_ttt_gui.run_gui(3, provided.PLAYERX, mc_move, NTRIALS, False)
