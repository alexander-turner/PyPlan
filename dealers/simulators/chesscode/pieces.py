import copy


class Piece:
    # Movement blueprints
    orthogonal = ((-1, 0), (0, -1), (1, 0), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    range = 8
    can_orthogonal = False  # indicates which movement directions are available
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
        actions = []
        for row_change, col_change in self.directions:
            new_position = copy.deepcopy(self.position)
            for i in range(self.range):
                new_position[0] += row_change
                new_position[1] += col_change
                if not board.in_bounds(new_position) or board.is_occupied(new_position):
                    if board.is_legal(Action(self, new_position)):
                        actions.append(copy.deepcopy(new_position))  # can move into enemy piece  TODO normal copy?
                    break
                else:
                    actions.append(copy.deepcopy(new_position))

        return (Action(self, action) for action in actions)  # so we know which piece is being moved

    # question is_legal function?

    def __str__(self):
        return self.abbreviation.upper() if self.color == 'white' else self.abbreviation


class Pawn(Piece):
    abbreviation = 'p'

    def get_actions(self, board):
        """Return legal actions for the given board."""
        movement_direction = board.movement_direction[self.color]
        actions = []

        # Move forward one if the square isn't occupied
        new_position = board.compute_position(self.position, (movement_direction, 0))
        if not board.is_occupied(new_position) and board.is_legal(Action(self, new_position)):
            actions.append(Action(self, new_position))

        # If we're in the initial position and the square two ahead is empty, we can move there
        new_position = board.compute_position(self.position, (movement_direction, 0))
        if not self.has_moved and not board.is_occupied(new_position) and \
           board.is_legal(Action(self, new_position)):
            actions.append(Action(self, new_position))

        # Check if enemy piece is at diagonals
        diagonals = ((movement_direction, -1), (movement_direction, 1))
        for diagonal in diagonals:
            new_position = board.compute_position(self.position, diagonal)
            if board.in_bounds(new_position) and board.is_occupied(new_position) and \
                    not board.is_same_color(self, new_position) and board.is_legal(Action(self, new_position)):
                actions.append(Action(self, new_position))

        return actions

    def append_if_valid(self, position_change, lst, board):
        """If the move to the given position is valid, append the resultant action to lst."""
        new_position = board.compute_position(self.position, position_change)
        new_action = Action(self, new_position)
        if board.is_legal(new_action):
            lst.append(new_action)


class Rook(Piece):
    can_orthogonal = True
    abbreviation = 'r'


class Knight(Piece):
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))
    abbreviation = 'n'

    def get_actions(self, board):
        moves = [Action(self, board.compute_position(self.position, delta)) for delta in self.deltas]
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

    def __init__(self, piece, new_position, special_type=None):
        """Initialize a Move object.

        :param piece: a pointer to the piece to be moved.
        :param new_position: the position to which the piece will move.
        :param special_type: what type (castling / en passant / pawn promotion) of special move, if any, this is.
        """
        self.piece = piece
        self.new_position = new_position
        self.special_type = special_type
