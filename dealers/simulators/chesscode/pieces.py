import numpy as np


class Piece:
    # Movement blueprints
    orthogonal = ((-1, 0), (0, -1), (1, 0), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    range = 8
    can_orthogonal = False  # which movement directions are available
    can_diagonal = False

    abbreviation = ''  # lower-case letter that represents the piece

    def __init__(self, position, color):
        self.position, self.initial_position = position, tuple(position)
        self.color = color

        self.directions = np.array([])
        if self.can_orthogonal:
            np.append(self.directions, self.orthogonal)
        if self.can_diagonal:
            np.append(self.directions, self.diagonal)

    def get_actions(self, board):
        actions = []
        for row_change, col_change in self.directions:
            new_position, change_unit = self.position, np.array([row_change, col_change])
            for _ in range(self.range):
                new_position += change_unit
                if board.is_legal(Action(self.position, new_position)):
                    actions.append(Action(self.position, new_position))
                elif not board.in_bounds(new_position) or board.is_occupied(new_position):
                    break

        return actions

    def __hash__(self):
        return hash(self.initial_position)

    def __str__(self):
        return self.abbreviation.upper() if self.color == 'white' else self.abbreviation


class Pawn(Piece):
    range = 1
    can_diagonal = True
    can_orthogonal = True
    abbreviation = 'p'

    def get_actions(self, board):
        """Return legal actions for the given board."""
        if not hasattr(self, 'movement_direction'):
            self.movement_direction = board.movement_direction[self.color]
        move_functions = [self.get_actions_one_step, self.get_actions_two_step, self.get_actions_diagonal,
                          self.get_actions_promotion, self.get_actions_en_passant]

        actions = [func(board) for func in move_functions]  # generate actions
        return [action for sublist in actions for action in sublist]  # flatten lists

    def get_actions_one_step(self, board):
        # Move forward one if the square isn't occupied
        new_position = self.position + np.array([self.movement_direction, 0])
        actions = []

        if board.in_bounds(new_position) and not board.is_occupied(new_position) and \
                board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        return actions

    def get_actions_two_step(self, board):
        # If we're in the initial position and the square two ahead is empty, we can move there
        new_position = self.position + np.array([self.movement_direction * 2, 0])
        actions = []

        if np.array_equal(self.position, self.initial_position) and not board.is_occupied(new_position) and \
           board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        return actions

    def get_actions_diagonal(self, board):
        # Check if enemy piece is at diagonals
        diagonals = ((self.movement_direction, -1), (self.movement_direction, 1))
        actions = []

        for diagonal in diagonals:
            new_position = self.position + np.array(diagonal)
            if board.in_bounds(new_position) and board.is_occupied(new_position) and \
                    board.is_legal(Action(self.position, new_position)):
                actions.append(Action(self.position, new_position))

        return actions

    def get_actions_promotion(self, board):
        # Pawn promotion
        enemy_back_line = board.bounds['top'] if board.movement_direction[self.color] == -1 else board.bounds['bottom']
        actions = []

        if self.position[0] == enemy_back_line:  # in enemy's back-line
            for promotion_piece in (Rook, Knight, Bishop, Queen):
                actions.append(Action(self.position, self.position, 'promotion', promotion_piece))

        return actions

    def get_actions_en_passant(self, board):
        actions = []

        for adjacent_position in self.position + np.array(((0, -1), (0, 1))):
            if board.in_bounds(adjacent_position) and board.last_action is not None and \
               np.array_equal(board.last_action.new_position, adjacent_position):
                adjacent_piece = board.get_piece(adjacent_position)  # retrieve the piece next to us
                last_action = board.last_action  # if last action was an enemy pawn moving two
                if isinstance(adjacent_piece, Pawn) and not board.is_same_color(self, adjacent_piece.position) and \
                        (last_action.new_position - last_action.current_position)[0] == 2:
                    diagonal = np.array((self.movement_direction, (adjacent_position - self.position)[1]))
                    actions.append(Action(self.position, self.position + diagonal, 'en passant'))

        return actions


class Rook(Piece):
    can_orthogonal = True
    abbreviation = 'r'


class Knight(Piece):
    range = 2
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))
    abbreviation = 'n'

    def get_actions(self, board):
        moves = [Action(self.position, self.position + np.array(delta)) for delta in self.deltas]
        return filter(board.is_legal, moves)


class Bishop(Piece):
    can_diagonal = True
    abbreviation = 'b'


class Queen(Piece):
    can_diagonal = True
    can_orthogonal = True
    abbreviation = 'q'


class King(Piece):
    range = 1
    can_diagonal = True
    can_orthogonal = True
    abbreviation = 'k'

    def get_actions(self, board):
        actions = super().get_actions(board)  # basic moves

        # Check to see if we can castle
        if np.array_equal(self.position, self.initial_position):
            for rook in [piece for piece in board.players[self.color].pieces if isinstance(piece, Rook)]:
                if np.array_equal(rook.position, rook.initial_position) and board.has_line_of_sight(self, rook.position):
                    side = (0, -1) if rook.position[1] < self.position[1] else (0, 1)  # if left of king
                    king_new_position = self.position + np.array((side[0], side[1]*2))
                    rook_new_position = self.position + np.array(side)
                    actions.append(Action(self.position, king_new_position, special_type='castle',
                                          special_params=Action(rook.position, rook_new_position)))
        return actions


class Action:
    def __init__(self, current_position, new_position, special_type=None, special_params=None):
        """Initialize a Move object.

        :param current_position: the starting piece position.
        :param new_position: the position to which the piece will move.
        :param special_type: what type (castling / en passant / pawn promotion) of special move, if any, this is.
        :param special_params: information required to execute the special move (ex: piece class for promotion)
        """
        self.current_position = current_position
        self.new_position = new_position
        self.special_type = special_type
        self.special_params = special_params

    @staticmethod
    def chess_notation(position):
        """Returns the chess notation (e.g. A1) of the given position."""
        row = 8 - position[0]
        col = chr(97 + position[1])
        return str(row) + str(col)

    def __str__(self):
        current, new = self.chess_notation(self.current_position), self.chess_notation(self.new_position)
        return current + " to " + new
