import copy
import operator
import dealers.simulators.chesscode.pieces as pieces  # rules and constructors for the pieces


class Board:
    """Initialize a chessboard."""
    height = 8
    width = 8
    bounds = {'top': 0, 'bottom': height - 1, 'left': 0, 'right': width - 1}
    movement_direction = {'white': -1, 'black': 1}  # which direction pawns of the given color can move

    piece_values = {'p': 1, 'n': 4, 'b': 3.5, 'r': 7, 'q': 13.5, 'k': 1000}  # values from Kurzdorfer 2003

    def __init__(self):
        self.current_state = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.players = {'white': Player(self, 'white'), 'black': Player(self, 'black')}
        for color in self.players:
            self.players[color].set_pieces()  # have the player set its pieces on the board

    def update_board(self, action):
        """Updates the board by taking the provided action (which is assumed to be legal).

        :return reward: the value of the piece taken; else 0.
        """
        piece = action[0]
        new_position = action[1]

        reward = 0
        if self.is_occupied(new_position):  # remove captured piece, if necessary
            removed_piece = self.current_state[new_position[0]][new_position[1]]
            self.players[removed_piece.color].pieces.remove(removed_piece)  # TODO fix board falling out of alignment with pieces (nonsensical moves)
            reward = self.piece_values[removed_piece.__str__().lower()]

        # Update board
        self.current_state[piece.position[0]][piece.position[1]] = ' '
        piece.position = new_position
        self.current_state[new_position[0]][new_position[1]] = piece

        return reward

    def is_legal(self, move):
        """Returns True if piece can take the action.

        :param move: a tuple (piece, new_position).
        """
        piece = move[0]
        start = piece.position  # where the move starts
        new_position = move[1]

        # Check whether the destination is in-bounds
        if not self.in_bounds(new_position):
            return False

        # Is not a knight (no LOS check for knights) and has clear LOS to target square
        if not isinstance(piece, pieces.Knight) and not self.has_line_of_sight(start, new_position):
            return False

        # If there's a piece at end, check if it's on same team / a king (who cannot be captured directly)
        if self.is_occupied(new_position) and (self.is_same_color(piece, new_position) or
                                               isinstance(self.current_state[new_position[0]][new_position[1]],
                                                          pieces.King)):
            return False

        return True

        # Be sure we aren't leaving our king in check
        sim_state = copy.deepcopy(self)
        sim_state.update_board(move)
        if sim_state.is_checked(piece.color, do_recurse=False):  # TODO fix - boolean param to avoid loops?
            return False

        return True

    def in_bounds(self, position):
        return self.bounds['top'] <= position[0] < self.bounds['bottom'] and\
               self.bounds['left'] <= position[1] < self.bounds['right']

    def is_occupied(self, position):
        return isinstance(self.current_state[position[0]][position[1]], pieces.Piece)

    def has_line_of_sight(self, start, new_position):
        """Returns true if the piece at start has a clear line of sight to its destination.

        Assumes that the move takes place on a line (whether it be horizontal, vertical, or diagonal).
        """
        current = copy.deepcopy(start)
        position_change = list(map(operator.sub, new_position, start))

        if position_change[0] != 0:
            row_change = 1 if position_change[0] > 0 else -1
        else:
            row_change = 0

        if position_change[1] != 0:
            col_change = 1 if position_change[1] > 0 else -1
        else:
            col_change = 0

        while True:
            # Move to next position along line.
            current[0] += row_change
            current[1] += col_change

            if current == new_position:  # simulate a do-while
                break

            if self.is_occupied(current):
                return False

        return True

    def is_same_color(self, piece, new_position):
        if not self.is_occupied(new_position):
            return False
        return piece.color == self.current_state[new_position[0]][new_position[1]].color

    def is_checked(self, color, do_recurse=True):
        """Returns true if a king of the given color would be checked at that position.

        :param color: the color of the king in question.
        :param do_recurse: whether we should ensure that the move wouldn't put the other king in check.
        """
        for piece in self.players[color].pieces:
            if isinstance(piece, pieces.King):
                king = piece

        enemy_color = 'black' if color == 'white' else 'white'
        for action in self.players[enemy_color].get_actions():  # TODO fix infinite recursion from get_actions
            if action[1] == king.position:  # if this action wouldn't leave enemy king in check
                if not do_recurse:  # don't bother simulating whether action will leave enemy king in check
                    return True
                sim_state = copy.deepcopy(self)
                sim_state.update_board(action)
                if not sim_state.is_checked(enemy_color, do_recurse=False):
                    return True

        return False

    @staticmethod
    def is_position(action, position):
        """Returns whether the action is at the given position."""
        return action[1] == position

    @staticmethod
    def compute_position(position, position_change):
        """Add (row, col) to (row_change, col_change)."""
        return list(map(sum, zip(position, position_change)))

    def __str__(self):
        board_str = ''
        for row in range(self.height):
            row_str = ''
            for col in range(self.width):
                row_str += self.current_state[row][col].__str__()
            board_str += row_str + '\n'
        return board_str


class Player:
    """A structure which stores the players and their pieces."""

    def __init__(self, board, color):
        self.board = board  # pointer to parent Board

        if color != 'white' and color != 'black':
            raise Exception('Invalid color - must be white or black.')
        self.color = color

        self.pieces = []

    def set_pieces(self):
        """Set the player's pieces in the correct location for their color."""
        is_white = self.color == 'white'
        if is_white:
            back_row = self.board.bounds['bottom']
        else:
            back_row = self.board.bounds['top']
        pawn_row = back_row + self.board.movement_direction[self.color]

        # Place the pawns
        for col in range(self.board.width):
            pawn = pieces.Pawn([pawn_row, col], self.color)
            self.pieces.append(pawn)
            self.board.current_state[pawn_row][col] = pawn

        # Place the back line
        for col in range(self.board.width):
            if col == 0 or col == 7:
                piece = pieces.Rook([back_row, col], self.color)
            elif col == 1 or col == 6:
                piece = pieces.Knight([back_row, col], self.color)
            elif col == 2 or col == 5:
                piece = pieces.Bishop([back_row, col], self.color)
            elif col == 3:
                piece = pieces.Queen([back_row, col], self.color)
            else:
                piece = pieces.King([back_row, col], self.color)
            self.pieces.append(piece)
            self.board.current_state[back_row][col] = piece

    # TODO cache moves after first generation, wipe when update_board called?
    # TODO Move class?
    def get_actions(self):  # TODO castling, pawn promotion, en passant
        actions = []  # list of tuples (piece, new_position)
        for piece in self.pieces:
            actions += piece.get_actions(self.board)
        return actions
