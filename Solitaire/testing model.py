from solitaire_base import SolitaireGame
from random import randint

game = SolitaireGame()
game.print_board()

def random_move(game):
    move = False
    while not move:
        from_column = randint(0, 8)
        to_column = randint(0, 8)
        move = game.move_logic(from_column, to_column)
    print(from_column, to_column)
    game.print_board()

random_move(game)
