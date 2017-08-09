import operator
import dealers.simulators.chesscode.pieces as pieces  # rules and constructors for the pieces


class Board:
    """Initialize a chessboard."""
    height = 8
    width = 8
    bounds = {'top': 0, 'bottom': height - 1, 'left': 0, 'right': width - 1}
    movement_direction = {'white': 1, 'black': -1}  # which direction pawns of the given color can move

    def __init__(self):
        self.current_state = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.players = {'white': Player(self, 'white'), 'black': Player(self, 'black')}
        for color in self.players:
            self.players[color].set_pieces()  # have the player set its pieces on the board

    def update_board(self, action):
        """Updates the board by taking the provided action (which is assumed to be legal)."""
        piece = action[0]
        new_position = action[1]
        self.current_state[piece.position[0]][piece.position[1]] = ' '
        self.current_state[new_position[0]][new_position[1]] = piece

    def is_occupied(self, position):
        return self.current_state[position[0]][position[1]] != ' '

    def in_bounds(self, position):
        return self.bounds['top'] <= position[0] < self.bounds['bottom'] and\
               self.bounds['left'] <= position[1] < self.bounds['right']

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
        if not self.is_occupied(new_position) and \
                (self.is_same_color(piece, new_position) or
                 isinstance(self.current_state[new_position[0]][new_position[1]], pieces.King)):
            return False

        # Be sure we aren't leaving our king in check
        if self.is_checked(piece.color, new_position):
            return False

        return True

    def has_line_of_sight(self, start, new_position):
        """Returns true if the piece at start has a clear line of sight to its destination.

        Assumes that the move takes place on a line (whether it be horizontal, vertical, or diagonal).
        """
        current = list(start)  # so we can modify values
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

            if not self.is_occupied(current):
                return False

            if current == new_position:  # simulate a do-while
                break

        return True

    def is_same_color(self, piece, new_position):
        if not self.is_occupied(piece.position) or not self.is_occupied(new_position):
            return False
        return piece.color == self.current_state[new_position[0]][new_position[1]].color

    def is_checked(self, color, position):
        """Returns true if a king of the given color would be checked at that position."""
        return False  # todo finish - simulate?

    def __str__(self):
        for row in range(self.height):
            row_str = ''
            for col in range(self.width):
                row_str += self.current_state[row][col]
            print(row_str)


class Player:
    """A structure which stores the players and their pieces."""

    def __init__(self, board, color):
        self.board = board  # pointer to parent Board

        if color != 'white' and color != 'black':
            raise Exception('Invalid color - must be white or black.')
        self.color = color

        self.pieces = {}  # hash by initial location
        self.is_checked = False  # whether our king is in check

    def set_pieces(self):
        """Set the player's pieces in the correct location for their color."""
        is_white = self.color == 'white'
        if is_white:
            back_row = self.board.bounds['top']
        else:
            back_row = self.board.bounds['bottom']
        pawn_row = back_row + self.board.movement_direction[self.color]

        # Place the pawns
        for col in range(self.board.width):
            pawn = pieces.Pawn([pawn_row, col], self.color)
            self.pieces[pawn] = pawn
            self.board.current_state[pawn_row][col] = pawn  # black pieces should be lowercase

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
            self.pieces[piece] = piece
            self.board.current_state[back_row][col] = piece

    def get_moves(self):  # TODO castling, pawn promotion, en passant
        actions = []  # list of tuples (piece, new_position)
        for piece in self.pieces:
            actions.append(piece.get_actions(self.board))
        return actions
