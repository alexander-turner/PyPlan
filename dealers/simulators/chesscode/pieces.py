import copy

class Piece:
    # Movement blueprints
    orthogonal = ((-1, 0), (0, -1), (1, 0), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    range = 8
    can_orthogonal = False  # indicates which movement directions are available to piece
    can_diagonal = False

    def __init__(self, position, color):
        self.position = position
        self.color = color

    def get_actions(self, board):
        directions = []
        if self.can_orthogonal:
            directions += self.orthogonal
        if self.can_diagonal:
            directions += self.diagonal

        actions = []
        for row_change, col_change in directions:
            new_position = copy.deepcopy(self.position)  # todo fix incorrect piece position
            for i in range(self.range):
                new_position[0] += row_change
                new_position[1] += col_change
                if not board.in_bounds(new_position) or board.is_occupied(new_position):
                    if board.is_legal((self, new_position)):
                        actions.append(copy.deepcopy(new_position))  # can move into enemy piece
                    break
                else:
                    actions.append(copy.deepcopy(new_position))

        return ((self, action) for action in actions)  # so we know which piece is being moved


class Pawn(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.start_row = position[0]

    def get_actions(self, board):
        """Return legal actions for the given board."""
        actions = []
        movement_direction = board.movement_direction[self.color]

        # Move forward one
        self.append_if_valid((movement_direction, 0), actions, board)

        # If we're in the initial position, we can move forward two
        if self.position[0] == self.start_row:
            self.append_if_valid((movement_direction * 2, 0), actions, board)

        # Check if enemy piece is at diagonals
        left_diagonal = (movement_direction, - 1)
        new_position = list(map(sum, zip(self.position, left_diagonal)))
        if board.in_bounds(new_position) and board.is_occupied(new_position) and \
                not board.is_same_color(self, new_position):
            self.append_if_valid(left_diagonal, actions, board)

        right_diagonal = (movement_direction, 1)
        new_position = list(map(sum, zip(self.position, right_diagonal)))
        if board.in_bounds(new_position) and board.is_occupied(new_position) and \
                board.is_same_color(self, new_position):
            self.append_if_valid(right_diagonal, actions, board)

        return actions

    def append_if_valid(self, position_change, lst, board):
        """If the move to the given position is valid, append the resultant action to lst."""
        new_position = list(map(sum, zip(self.position, position_change)))
        new_action = (self, new_position)
        if board.is_legal(new_action):
            lst.append(new_action)

    def __str__(self):
        return 'P' if self.color == 'white' else 'p'


class Rook(Piece):
    can_orthogonal = True

    def __str__(self):
        return 'R' if self.color == 'white' else 'r'


class Knight(Piece):
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))

    def get_actions(self, board):
        moves = [(self, delta) for delta in self.deltas]
        return filter(board.is_legal, moves)

    def __str__(self):
        return 'N' if self.color == 'white' else 'n'


class Bishop(Piece):
    can_diagonal = True

    def __str__(self):
        return 'B' if self.color == 'white' else 'b'


class Queen(Piece):
    can_diagonal = True
    can_orthogonal = True

    def __str__(self):
        return 'Q' if self.color == 'white' else 'q'


class King(Piece):
    range = 1
    can_diagonal = True
    can_orthogonal = True

    def __str__(self):
        return 'K' if self.color == 'white' else 'k'
