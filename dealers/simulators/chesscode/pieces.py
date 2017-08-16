import copy


class Piece:
    # Movement blueprints
    orthogonal = ((-1, 0), (0, -1), (1, 0), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    range = 8
    can_orthogonal = False  # which movement directions are available
    can_diagonal = False

    has_moved = False  # whether the piece has moved in this game
    abbreviation = ''  # lower-case letter that represents the piece

    def __init__(self, position, color):
        self.position = position
        self.color = color

        self.directions = []
        if self.can_orthogonal:
            self.directions += self.orthogonal
        if self.can_diagonal:
            self.directions += self.diagonal

    def get_actions(self, board):
        positions = []
        for row_change, col_change in self.directions:
            new_position = copy.deepcopy(self.position)
            for _ in range(self.range):
                new_position = board.compute_position(new_position, (row_change, col_change))
                if board.is_legal(Action(self.position, new_position)):
                    positions.append(copy.deepcopy(new_position))
                elif not board.in_bounds(new_position) or board.is_occupied(new_position):
                    break

        return (Action(self.position, position) for position in positions)

    def __str__(self):
        return self.abbreviation.upper() if self.color == 'white' else self.abbreviation


class Pawn(Piece):
    range = 1
    can_diagonal = True
    can_orthogonal = True
    abbreviation = 'p'

    def get_actions(self, board):
        """Return legal actions for the given board."""
        movement_direction = board.movement_direction[self.color]
        actions = []

        # Move forward one if the square isn't occupied
        new_position = board.compute_position(self.position, (movement_direction, 0))
        if board.in_bounds(new_position) and not board.is_occupied(new_position) and \
                board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        # If we're in the initial position and the square two ahead is empty, we can move there
        new_position = board.compute_position(self.position, (movement_direction * 2, 0))
        if not self.has_moved and not board.is_occupied(new_position) and \
           board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        # Check if enemy piece is at diagonals
        diagonals = ((movement_direction, -1), (movement_direction, 1))
        for diagonal in diagonals:
            new_position = board.compute_position(self.position, diagonal)
            if board.in_bounds(new_position) and board.is_occupied(new_position) and \
                    board.is_legal(Action(self.position, new_position)):
                actions.append(Action(self.position, new_position))

        enemy_back_line = board.bounds['top'] if board.movement_direction[self.color] == -1 else board.bounds['bottom']
        if self.position[0] == enemy_back_line:  # in enemy's back-line
            for promotion_piece in (Rook, Knight, Bishop, Queen):
                actions.append(Action(self.position, self.position, 'promotion', promotion_piece))

        return actions


class Rook(Piece):
    can_orthogonal = True
    abbreviation = 'r'


class Knight(Piece):
    range = 2
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))
    abbreviation = 'n'

    def get_actions(self, board):
        moves = [Action(self.position, board.compute_position(self.position, delta)) for delta in self.deltas]
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


class Action:
    has_moved = False  # whether this piece has moved from its starting position

    def __init__(self, current_position, new_position, special_type=None, special_params=None):
        """Initialize a Move object.

        :param current_position: the starting piece position.
        :param new_position: the position to which the piece will move.
        :param special_type: what type (castling / en passant / pawn promotion) of special move, if any, this is.
        :param special_params: information on the special move (ex: which side to castle)
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
