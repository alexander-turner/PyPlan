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

        for x, y in directions:
            collision = False
            new_position = self.position
            for i in range(1, self.range + 1):
                new_position[0] += x
                new_position[1] += y
                if collision or not board.in_bounds(new_position):
                    break
                if board.is_occupied(new_position):
                    collision = True
                    if board.is_legal((self, new_position)):
                        actions.append(new_position)  # can move into enemy piece
                else:
                    actions.append(new_position)

        actions = filter(board.in_bounds, actions)

        return ((self, action) for action in actions)  # so we know which piece is being moved

    def __str__(self):
        raise NotImplementedError


class Pawn(Piece):
    def __init__(self, position, color):
        super().__init__(position, color)
        self.start_row = position[0]

    def get_actions(self, board):
        """Return legal actions for the given board."""
        actions = []
        movement_direction = board.movement_direction[self.color]

        # Move forward one
        actions.append(process_action(board, self, (movement_direction, 0)))

        # If we're in the initial position, we can move forward two
        if self.position[0] == self.start_row:
            actions.append(process_action(board, self, (movement_direction * 2, 0)))

        # Check if enemy piece is at diagonals
        left_diagonal = (self.position[0] + movement_direction, self.position[1] - 1)
        if board.in_bounds(left_diagonal) and board.is_occupied(left_diagonal) and \
                not board.is_same_color(self, left_diagonal):
            actions.append(process_action(board, self, left_diagonal))

        right_diagonal = (self.position[0] + movement_direction, self.position[1] + 1)
        if board.in_bounds(right_diagonal) and board.is_occupied(right_diagonal) and \
                board.is_same_color(self, right_diagonal):
            actions.append(process_action(board, self, right_diagonal))

        return actions

    def __str__(self):
        return 'P' if self.color == 'white' else 'p'


class Rook(Piece):
    can_orthogonal = True

    def __init__(self, position, color):
        super().__init__(position, color)
        self.can_castle = True  # whether the rook has moved or castled already

    def __str__(self):
        return 'R' if self.color == 'white' else 'r'


class Knight(Piece):
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))

    def get_actions(self, board):
        moves = ((self, delta) for delta in self.deltas)
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


def process_action(board, piece, position_change):
    """Handle creation and checking of new action. Returns the action if valid; else, returns None."""
    new_position = list(map(sum, zip(piece.position, position_change)))
    new_action = (piece, new_position)
    if board.is_legal(new_action):
        return new_action

