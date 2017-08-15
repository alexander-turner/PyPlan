import copy
import operator
import dealers.simulators.chesscode.pieces as pieces  # rules and constructors for the pieces
from functools import partial


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

        # Whether is_legal() should verify that our king is not put in check by actions
        self.verify_not_checked = True  # only sim_states at 1st-level recursion should have this as False

    def update_board(self, action):
        """Updates the board by taking the provided action (which is assumed to be legal).

        :return reward: the value of the piece taken; else 0.
        """
        reward = 0
        piece = self.get_piece(action.current_position)

        if self.is_occupied(action.new_position):  # remove captured piece, if necessary
            removed_piece = self.get_piece(action.new_position)
            self.players[removed_piece.color].pieces.remove(removed_piece)
            reward = self.piece_values[removed_piece.abbreviation]

        # Update board
        self.set_piece(action.current_position, ' ')
        piece.position = action.new_position
        self.set_piece(action.new_position, piece)

        return reward

    def is_legal(self, action):
        """Returns True if piece can take the action.

        :param action: a tuple (piece, new_position).
        :param verify_not_checked: whether we need to see if this move would leave our king in check.
        """
        # Check whether the destination is in-bounds
        if not self.in_bounds(action.new_position):
            return False

        # If has LOS to the new position
        piece = self.get_piece(action.current_position)
        if not self.has_line_of_sight(piece, action.new_position):
            return False

        # If there's a piece at end, check if it's on same team / a king (who cannot be captured directly)
        if self.is_occupied(action.new_position) and (self.is_same_color(piece, action.new_position) or
                                                      isinstance(self.get_piece(action.new_position), pieces.King)):
            return False

        # Be sure we aren't leaving our king in check
        if self.verify_not_checked:  # todo king can move into check
            sim_state = copy.deepcopy(self)
            sim_state.update_board(action)
            sim_state.verify_not_checked = False
            if sim_state.is_checked(piece.color):
                return False

        return True

    def in_bounds(self, position):
        return self.bounds['top'] <= position[0] < self.bounds['bottom'] and \
               self.bounds['left'] <= position[1] < self.bounds['right']

    def is_occupied(self, position):
        return isinstance(self.get_piece(position), pieces.Piece)

    def get_piece(self, position):  # question check in-bounds?
        return self.current_state[position[0]][position[1]]

    def set_piece(self, position, new_value):
        self.current_state[position[0]][position[1]] = new_value

    def has_line_of_sight(self, piece, new_position):
        """Returns true if the piece has a clear line of sight to its destination."""
        # LOS concerns don't apply to knights
        if isinstance(piece, pieces.Knight):  # todo pawns can't move diagonally
            return True

        # Check that the move is a valid line
        position_change = self.compute_change(piece.position, new_position)  # total change in position
        if position_change[0] != 0 and position_change[1] != 0:
            if not piece.can_diagonal or abs(position_change[0]) != abs(position_change[1]):
                return False
        elif not isinstance(piece, pieces.Pawn) and not piece.can_orthogonal:  # orthogonal move but can't do that
            return False

        row_change, col_change = 0, 0
        if position_change[0] != 0:
            row_change = 1 if position_change[0] > 0 else -1  # per-iteration row change

        if position_change[1] != 0:
            col_change = 1 if position_change[1] > 0 else -1  # per-iteration col change

        position = copy.deepcopy(piece.position)
        while True:
            position = self.compute_position(position, [row_change, col_change])
            if position == new_position:  # simulate a do-while
                return True
            if self.is_occupied(position):
                return False

    def is_same_color(self, piece, new_position):
        """Returns True if the pieces are of the same color."""
        if not self.is_occupied(new_position):
            return False
        return piece.color == self.get_piece(new_position).color

    def is_checked(self, color):
        """Returns true if the king of the given color is in check.

        :param color: the king's color.
        """
        for piece in self.players[color].pieces:
            if isinstance(piece, pieces.King):
                king = piece
                break

        enemy_color = 'black' if color == 'white' else 'white'
        partial_in_range = partial(self.in_range, new_position=king.position)
        filtered_pieces = filter(partial_in_range, self.players[enemy_color].pieces)

        partial_has_line_of_sight = partial(self.has_line_of_sight, new_position=king.position)
        filtered_pieces = filter(partial_has_line_of_sight, filtered_pieces)

        actions = []
        for piece in filtered_pieces:
            actions += piece.get_actions(self)
        return any(action.new_position == king.position for action in actions)

    def in_range(self, piece, new_position):
        """Returns True if the piece could potentially move to the given position (i.e. within movement bounds)."""
        position_change = self.compute_change(piece.position, new_position)
        return -1 * piece.range <= position_change[0] <= piece.range and \
               -1 * piece.range <= position_change[1] <= piece.range

    @staticmethod
    def compute_position(current_position, position_change):
        """Add (row, col) to (row_change, col_change)."""
        return list(map(operator.add, current_position, position_change))

    @staticmethod
    def compute_change(current_position, new_position):
        return list(map(operator.sub, new_position, current_position))

    def __str__(self):
        board_str = ''
        for row in range(self.height):
            row_str = ''
            for col in range(self.width):
                row_str += self.get_piece((row, col)).__str__()
            board_str += row_str + '\n'
        return board_str


class Player:
    """A structure which stores the players and their pieces."""

    def __init__(self, board, color):
        self.board = board  # pointer to parent Board

        if color not in ('white', 'black'):
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
            position = [pawn_row, col]
            pawn = pieces.Pawn(position, self.color)
            self.pieces.append(pawn)
            self.board.set_piece(position, pawn)

        # Place the back line
        for col in range(self.board.width):
            position = [back_row, col]
            if col == 0 or col == 7:
                piece = pieces.Rook(position, self.color)
            elif col == 1 or col == 6:
                piece = pieces.Knight(position, self.color)
            elif col == 2 or col == 5:
                piece = pieces.Bishop(position, self.color)
            elif col == 3:
                piece = pieces.Queen(position, self.color)
            else:
                piece = pieces.King(position, self.color)
            self.pieces.append(piece)
            self.board.set_piece(position, piece)

    # TODO cache moves after first generation, wipe when update_board called?
    def get_actions(self):  # TODO castling, pawn promotion, en passant
        actions = []  # list of tuples (piece, new_position)
        for piece in self.pieces:
            actions += piece.get_actions(self.board)
        return actions
