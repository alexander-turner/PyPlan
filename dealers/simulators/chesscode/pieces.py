import copy


class Piece:
    # Movement blueprints
    orthogonal = ((-1, 0), (0, -1), (1, 0), (0, 1))
    diagonal = ((-1, -1), (-1, 1), (1, -1), (1, 1))

    range = 8
    can_orthogonal = False  # which movement directions are available
    can_diagonal = False

    abbreviation = ''  # lower-case letter that represents the piece

    def __init__(self, position, color):
        self.position = position
        self.color = color

        self.has_moved = False
        self.directions = []
        if self.can_orthogonal:
            self.directions += self.orthogonal
        if self.can_diagonal:
            self.directions += self.diagonal

    def get_actions(self, board):
        actions = []
        for row_change, col_change in self.directions:
            new_position = self.position
            for _ in range(self.range):
                new_position = board.compute_position(new_position, (row_change, col_change))
                if board.is_legal(Action(self.position, new_position)):
                    actions.append(Action(self.position, new_position))
                elif not board.in_bounds(new_position) or board.is_occupied(new_position):
                    break

        return actions

    def __str__(self):
        return self.abbreviation.upper() if self.color == 'white' else self.abbreviation


class Pawn(Piece):
    range = 1
    can_diagonal = True
    can_orthogonal = True
    abbreviation = 'p'

    def get_actions(self, board):
        """Return legal actions for the given board."""
        move_functions = [self.get_actions_one_step, self.get_actions_two_step, self.get_actions_diagonal,
                          self.get_actions_promotion, self.get_actions_en_passant]

        action_lists = [func(board) for func in move_functions]  # generate actions
        return [action for sublist in action_lists for action in sublist]  # flatten lists

    def get_actions_one_step(self, board):
        # Move forward one if the square isn't occupied
        new_position = board.compute_position(self.position, (board.movement_direction[self.color], 0))
        actions = []

        if board.in_bounds(new_position) and not board.is_occupied(new_position) and \
                board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        return actions

    def get_actions_two_step(self, board):
        # If we're in the initial position and the square two ahead is empty, we can move there
        new_position = board.compute_position(self.position, (board.movement_direction[self.color] * 2, 0))
        actions = []

        if not self.has_moved and not board.is_occupied(new_position) and \
           board.is_legal(Action(self.position, new_position)):
            actions.append(Action(self.position, new_position))

        return actions

    def get_actions_diagonal(self, board):
        # Check if enemy piece is at diagonals
        diagonals = ((board.movement_direction[self.color], -1), (board.movement_direction[self.color], 1))
        actions = []

        for diagonal in diagonals:
            new_position = board.compute_position(self.position, diagonal)
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

        for adjacent_position in [board.compute_position(self.position, side) for side in ((0, -1), (0, 1))]:
            if board.in_bounds(adjacent_position) and board.last_action is not None and \
               board.last_action.new_position == adjacent_position:
                adjacent_piece = board.get_piece(adjacent_position)  # retrieve the piece next to us

                # If last action was an enemy pawn moving two
                if isinstance(adjacent_piece, Pawn) and not board.is_same_color(self, adjacent_piece.position) and \
                   board.compute_change(board.last_action.current_position, board.last_action.new_position)[0] == 2:
                    diagonal = (board.movement_direction[self.color],
                                board.compute_change(self.position, adjacent_position)[1])
                    new_position = board.compute_position(self.position, diagonal)
                    actions.append(Action(self.position, new_position, 'en passant'))

        return actions


class Rook(Piece):
    can_orthogonal = True
    abbreviation = 'r'


class Knight(Piece):
    range = 2
    deltas = ((2, 1), (2, -1), (-2, -1), (-2, 1), (1, 2), (1, -2), (-1, 2), (-1, -2))
    abbreviation = 'n'

    def get_actions(self, board):
        actions = [Action(self.position, board.compute_position(self.position, delta)) for delta in self.deltas]
        return filter(board.is_legal, actions)


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
        if not self.has_moved:
            for rook in [piece for piece in board.get_pieces(self.color) if isinstance(piece, Rook)]:  # TODO implement
                if not rook.has_moved and board.has_line_of_sight(self, rook.position):
                    side = (0, -1) if rook.position[1] < self.position[1] else (0, 1)  # if left of king
                    king_new_position = board.compute_position(self.position, (side[0], side[1]*2))
                    rook_new_position = board.compute_position(self.position, side)
                    actions.append(Action(self.position, king_new_position, special_type='castle',
                                          special_params=Action(rook.position, rook_new_position)))
        return actions


class Action:
    def __init__(self, current_position, new_position, special_type=None, special_params=None):
        """
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
        return self.chess_notation(self.current_position) + " to " + self.chess_notation(self.new_position)
