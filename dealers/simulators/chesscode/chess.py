import dealers.simulators.chesscode.pieces as pieces
from functools import partial
from itertools import chain


class Board:
    """Initialize a chessboard."""
    height = 8
    width = 8
    bounds = {'top': 0, 'bottom': height - 1, 'left': 0, 'right': width - 1}
    movement_direction = {'white': -1, 'black': 1}  # which direction pawns of the given color can move

    piece_values = {'p': 1, 'n': 4, 'b': 3.5, 'r': 7, 'q': 13.5, 'k': 1000}  # values from Kurzdorfer 2003

    def __init__(self):
        self.current_state = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.pieces = {'white': {}, 'black': {}}
        self.kings = {}

        self.cached_actions = []  # should be updated by an outside simulator each time the board changes
        self.last_action = None  # the last action executed

        # Whether is_legal() should verify that our king is not put in check by actions
        self.verify_not_checked = True  # only sim_states at 1st-level recursion should have this as False
        self.allow_king_capture = False  # used to see if moves are checking the king

    def set_pieces(self):
        """Set up the initial piece configuration."""
        for color in ('white', 'black'):
            back_row = self.bounds['bottom'] if color == 'white' else self.bounds['top']
            pawn_row = back_row + self.movement_direction[color]

            for col in range(self.width):
                # Set the pawn row
                pawn = pieces.Pawn([pawn_row, col], color)
                self.set_piece([pawn_row, col], pawn)
                self.pieces[color][pawn] = pawn

                # Set the back row
                if col == 0 or col == 7:
                    piece = pieces.Rook([back_row, col], color)
                elif col == 1 or col == 6:
                    piece = pieces.Knight([back_row, col], color)
                elif col == 2 or col == 5:
                    piece = pieces.Bishop([back_row, col], color)
                elif col == 3:
                    piece = pieces.Queen([back_row, col], color)
                else:
                    piece = pieces.King([back_row, col], color)
                    self.kings[color] = piece
                self.set_piece([back_row, col], piece)
                self.pieces[color][piece] = piece

    def get_actions(self, color):
        """Generate all actions available for pieces of the given color."""
        return list(chain.from_iterable(piece.get_actions(self) for piece in self.pieces[color]))

    def is_legal(self, action):
        if not self.in_bounds(action.new_position):
            return False

        piece = self.get_piece(action.current_position)
        if not self.has_line_of_sight(piece, action.new_position):
            return False

        # If there's a piece at end, check if it's on same team / a king (who cannot be captured directly)
        if self.is_occupied(action.new_position) and \
                (self.is_same_color(piece, action.new_position) or
                 (not self.allow_king_capture and isinstance(self.get_piece(action.new_position), pieces.King))):
            return False

        # Be sure we aren't leaving our king in check
        if self.verify_not_checked:
            # Simulate taking the action
            piece_to_remove = self.get_piece(action.new_position)
            self.move_piece(action)

            # See if our king is in check
            self.verify_not_checked = False
            to_return = not self.is_checked(piece.color)

            # Undo the action
            self.verify_not_checked = True
            self.move_piece(pieces.Action(action.new_position, action.current_position))
            self.set_piece(action.new_position, piece_to_remove)
            if piece_to_remove != ' ':
                self.pieces[piece_to_remove.color][piece_to_remove] = piece_to_remove
            return to_return

        return True

    def in_bounds(self, position):
        return self.bounds['top'] <= position[0] <= self.bounds['bottom'] and \
               self.bounds['left'] <= position[1] <= self.bounds['right']

    def is_occupied(self, position):
        return self.get_piece(position) != ' '

    def has_line_of_sight(self, piece, new_position):
        """Returns true if the piece has a clear line of sight to its destination."""
        # LOS doesn't apply to knights
        if isinstance(piece, pieces.Knight):
            return True

        # Check that the move is a valid line
        position_change = self.compute_change(piece.position, new_position)  # total change in position
        if position_change[0] != 0 and position_change[1] != 0:  # moving at least one square in each direction
            if not piece.can_diagonal or abs(position_change[0]) != abs(position_change[1]):
                return False
        elif not piece.can_orthogonal:  # orthogonal move but can't do that
            return False

        row_change, col_change = 0, 0
        if position_change[0] != 0:
            row_change = 1 if position_change[0] > 0 else -1  # per-iteration row change

        if position_change[1] != 0:
            col_change = 1 if position_change[1] > 0 else -1  # per-iteration col change

        position = piece.position
        while True:
            position = self.compute_position(position, [row_change, col_change])
            if position == new_position:  # simulate a do-while
                return True
            if self.is_occupied(position):
                return False

    def is_same_color(self, piece, new_position):
        """Returns True if the pieces are of the same color. Assumes new_position is occupied by a piece."""
        return piece.color == self.get_piece(new_position).color

    def is_checked(self, color):
        """Returns true if the king of the given color is in check.

        :param color: the king's color.
        """
        enemy_pieces = self.pieces['black' if color == 'white' else 'white']
        partial_in_range = partial(self.in_range, new_position=self.kings[color].position)
        filtered_pieces = filter(partial_in_range, enemy_pieces)

        partial_has_line_of_sight = partial(self.has_line_of_sight, new_position=self.kings[color].position)
        filtered_pieces = filter(partial_has_line_of_sight, filtered_pieces)

        self.allow_king_capture = True
        actions = chain.from_iterable([piece.get_actions(self) for piece in filtered_pieces])
        self.allow_king_capture = False
        return any(action.new_position == self.kings[color].position for action in actions)

    def in_range(self, piece, new_position):
        """Returns True if the piece could potentially move to the given position (i.e. within movement bounds)."""
        position_change = self.compute_change(piece.position, new_position)
        return -1 * piece.range <= position_change[0] <= piece.range and \
               -1 * piece.range <= position_change[1] <= piece.range

    def get_piece(self, position):
        return self.current_state[position[0]][position[1]]

    def set_piece(self, position, new_value):
        self.current_state[position[0]][position[1]] = new_value

    def move_piece(self, action):
        """Move the piece at action.current_position, returning the reward for capturing a piece (if applicable)."""
        piece = self.get_piece(action.current_position)
        reward = self.remove_piece(action.new_position)

        # Update piece information
        self.set_piece(action.current_position, ' ')
        piece.position = action.new_position
        piece.has_moved = True

        if action.special_type == 'promotion':  # pawn promotion
            piece = action.special_params(action.current_position, piece.color)
            self.pieces[piece.color][piece] = piece
            # Take the difference since the promoted piece is more valuable than a pawn
            reward = self.piece_values[piece.abbreviation] - self.piece_values['p']
        elif action.special_type == 'en passant':
            reward = self.remove_piece(self.last_action.new_position)
        elif action.special_type == 'castle':
            self.move_piece(action.special_params)  # contains rook's action

        self.set_piece(action.new_position, piece)

        return reward

    def remove_piece(self, position):
        """Remove the piece from the corresponding piece set and return its value."""
        reward = 0
        if self.is_occupied(position):  # remove captured piece, if necessary
            piece = self.get_piece(position)
            del self.pieces[piece.color][piece]
            reward = self.piece_values[piece.abbreviation]
        self.set_piece(position, ' ')
        return reward

    @staticmethod
    def compute_position(current_position, position_change):
        """Add (row, col) to (row_change, col_change)."""
        return [current_position[0] + position_change[0], current_position[1] + position_change[1]]

    @staticmethod
    def compute_change(current_position, new_position):
        return [new_position[0] - current_position[0], new_position[1] - current_position[1]]

    def __copy__(self):
        board = Board()
        for row in range(board.height):  # copy each piece
            for col in range(board.width):
                piece = self.get_piece((row, col))
                if piece != ' ':
                    new_piece = piece.copy()
                    board.set_piece((row, col), new_piece)
                    board.pieces[new_piece.color][new_piece] = new_piece
                    if isinstance(new_piece, pieces.King):
                        board.kings[new_piece.color] = new_piece
                else:
                    board.set_piece((row, col), piece)

        board.cached_actions, board.last_action = self.cached_actions, self.last_action
        board.verify_not_checked, board.allow_king_capture = self.verify_not_checked, self.allow_king_capture

        return board

    def __str__(self):
        board_str = ''
        for row in range(self.height):
            row_str = ''.join([self.get_piece((row, col)).__str__() for col in range(self.width)])
            board_str += row_str + '\n'
        return board_str
