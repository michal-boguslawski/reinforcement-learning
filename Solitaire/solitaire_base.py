import numpy as np

np.set_printoptions(linewidth=200)


def prepare_input(a, b, c):
    if a.ndim != 2:
        a = np.reshape(a, (-1, 2))
    if b.ndim != 2:
        b = np.reshape(b, (-1, 2))
    if c.ndim != 2:
        c = np.reshape(c, (-1, 2))
    return np.concatenate([a, b, c])


class Foundation:
    def __init__(self):
        self.foundation = np.zeros(4, dtype=np.uint32)

    def get_color(self, color):
        return self.foundation[color - 1]

    def set_color(self, color, figure):
        if color == 5 or color == 0:
            return False
        elif self.foundation[color - 1] + 1 == figure:
            self.foundation[color - 1] = figure
            return True
        return False

    def reset(self):
        self.foundation = np.zeros(4, dtype=np.uint32)
        return True

    def check_win(self):
        if (self.foundation == 13).all():
            return True
        return False


class Waste:
    def __init__(self):
        self.waste = []
        self.position = 0

    def start(self, cards):
        self.waste = cards.tolist()
        self.position = 0
        return True

    def get_card(self):
        if self.get_len() == 0:
            return False
        return self.waste[self.position]

    def change_card(self):
        waste_len = self.get_len()
        if waste_len == 0:
            return False
        self.position += 1
        self.position %= waste_len
        return True

    def remove_card(self):
        card = self.waste.pop(self.position)
        self.position -= 1
        self.position = max(self.position, 0)
        return card

    def get_len(self):
        return len(self.waste)


class Tableau:
    def __init__(self):
        self.board = np.zeros((20, 7, 2), dtype=np.uint32)
        self.positions = np.arange(7)

    def start(self, cards):
        self.positions = np.arange(7)
        board = np.zeros((20, 7, 2), dtype=np.uint32)
        step = 0
        for col in range(7):
            for row in range(col + 1):
                board[row, col] = cards[step]
                step += 1
        self.board = board
        return True

    def put_card(self, card, column):
        face_index, face_card = self.get_face_card(column)
        face_card_color, face_card_figure = face_card
        card_color, card_figure = card
        if (face_card_color % 2 != card_color % 2 and face_card_figure - 1 == card_figure) or \
                (face_card_color == 0 and card_figure == 13):
            self.board[face_index, column] = card
            return True
        return False

    def get_face_card(self, column):
        card_column = self.board[:, column]
        face_index = np.argmax(card_column[:, 0] == 0)
        card = card_column[face_index - 1]
        return face_index, card

    def remove_face_card(self, column):
        face_index, face_card = self.get_face_card(column)
        self.board[face_index - 1, column] = 0
        if face_index - 1 == self.positions[column]:
            self.positions[column] -= 1
        return True

    def move_from_column_to_column(self, from_column, to_column):
        from_column_face_index, _ = self.get_face_card(from_column)
        to_column_face_index, to_column_face_card = self.get_face_card(to_column)
        from_position = max(self.positions[from_column], 0)
        board = self.board
        from_column_cards = board[from_position:from_column_face_index, from_column]
        condition = (
                (to_column_face_card[0] % 2 != from_column_cards[:, 0] % 2) &
                (to_column_face_card[1] - 1 == from_column_cards[:, 1])
        )
        if condition.any():
            from_move_index = np.argmax(condition)
            moved_cards = from_column_cards[from_move_index:]
            self.board[to_column_face_index:(to_column_face_index + len(moved_cards)), to_column] = moved_cards
            self.board[(from_position + from_move_index):from_column_face_index, from_column] = 0
            if from_move_index == 0:
                self.positions[from_column] -= 1
            return True
        elif len(from_column_cards) == 0:
            return False
        elif board[0, to_column, 0] == 0 and from_column_cards[0, 1] == 13:
            self.board[0:len(from_column_cards), to_column] = from_column_cards
            self.board[from_position:from_column_face_index, from_column] = 0
            self.positions[from_column] -= 1
            return True

        return False


class SolitaireGame:
    '''
    game = SolitaireGame()
    game.start_game()
    game.tableau.board[:, :, 0]
    game.tableau.move_from_column_to_column()
    game.move(7, 7, True)
    '''
    def __init__(self, max_steps: int = 1000):
        self.count_moves = 0
        self.max_steps = max_steps
        self.waste = Waste()
        self.foundation = Foundation()
        self.tableau = Tableau()
        colors = {
            0: 'empty',
            1: 'hearts',
            3: 'diamonds',
            2: 'clubs',
            4: 'spades',
            5: 'not visible'
        }
        self.colors = colors
        figures = {
            0: 'empty',
            1: 'Ace',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: '10',
            11: 'Jack',
            12: 'Queen',
            13: 'King',
            14: 'not visible'
        }
        self.figures = figures
        self.reset_board()
        self.color_vectorizer = np.vectorize(colors.get)
        self.figure_vectorizer = np.vectorize(figures.get)

    def reset_board(self):
        self.count_moves = 0
        colors = np.arange(4, dtype=np.uint32)
        figures = np.arange(13)
        full_stack = np.array(np.meshgrid(colors, figures)).T.reshape(-1, 2) + 1
        np.random.shuffle(full_stack)
        _ = self.foundation.reset()
        _ = self.tableau.start(full_stack)
        _ = self.waste.start(full_stack[-24:])
        return True

    def print_board(self):
        positions = self.tableau.positions
        arr = np.empty((20, 7), dtype='U32')
        board = self.tableau.board
        max_row = np.argmin(board[:, :, 0] != 0, 0).max()
        arr += self.figure_vectorizer(board[:, :, 1])
        arr += self.color_vectorizer(board[:, :, 0])
        arr[arr == 'emptyempty'] = ''
        for col, pos in enumerate(positions):
            arr[:max([pos, 0]), col] = '*'
        waste_card = self.waste.get_card()
        if waste_card:
            print('Waste:', self.figures[waste_card[1]] + self.colors[waste_card[0]])
        else:
            print('Waste: empty')
        foundation = self.figure_vectorizer(self.foundation.foundation)
        foundation = foundation.astype('U32')
        print('Foundation:', foundation + self.color_vectorizer(np.arange(1, 5)))
        arr = np.array2string(arr[:max_row], formatter={'all': lambda x: f"{x:12}"}, separator=' ')
        print(arr)

    def print_to_nn(self):
        board = self.tableau.board.copy()
        positions = self.tableau.positions
        for col, pos in enumerate(positions):
            board[:max([pos, 0]), col] = np.array([5, 14])
        foundation = self.foundation.foundation.copy()
        foundation = np.stack([np.arange(1, 5), foundation], axis=1)
        waste_card = self.waste.get_card()
        if waste_card:
            return prepare_input(board, foundation, np.array(waste_card))
        return prepare_input(board, foundation, np.zeros(2, dtype=np.uint32))

    def move(self, from_column: int, to_column: int, game_type: str, if_print_at_end: bool = True):
        value = self.move_logic(from_column, to_column)
        if game_type == 'manual':
            if self.foundation.check_win():
                print('Win')
                self.reset_board()
            if not value:
                print('Zły ruch')
            self.print_board()
        elif game_type == 'nn':
            board = self.print_to_nn()
            win = self.foundation.check_win()
            self.count_moves += 1
            termination = self.count_moves > self.max_steps
            if win or termination:
                if if_print_at_end:
                    self.print_board()
                self.reset_board()
            return board, value, win, termination

    def move_logic(self, from_column: int, to_column: int):
        # change card in waste
        if from_column == 7 and to_column == 7:
            return self.waste.change_card()
        # move card from waste to foundation
        elif from_column == 7 and to_column == 8:
            waste_card = self.waste.get_card()
            if waste_card:
                if self.foundation.set_color(waste_card[0], waste_card[1]):
                    _ = self.waste.remove_card()
                    return True
            return False
        # move card from tableau to foundation
        elif from_column < 7 and to_column == 8:
            _, face_card = self.tableau.get_face_card(from_column)
            if self.foundation.set_color(face_card[0], face_card[1]):
                return self.tableau.remove_face_card(from_column)
            return False
        # move card from tableau to tableau
        elif from_column < 7 and to_column < 7:
            return self.tableau.move_from_column_to_column(from_column, to_column)
        elif from_column == 7 and to_column < 7:
            waste_card = self.waste.get_card()
            if waste_card:
                if self.tableau.put_card(waste_card, to_column):
                    _ = self.waste.remove_card()
                    return True
            return False

        else:
            return False

    def start_game(self):
        self.print_board()
        while 1 == 1:
            move = input('Select move: ')
            if move == 'exit':
                break
            from_column = int(move[0])
            to_column = int(move[1])
            print('From: ', from_column, 'To:', to_column)
            self.move(from_column, to_column, 'manual')
