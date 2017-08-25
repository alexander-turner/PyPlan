import copy
import numpy as np
import os
import pygame
from abstract import abstract_state
from dealers.simulators.chesscode import chess


class ChessState(abstract_state.AbstractState):
    env_name = "Chess"
    num_players = 2

    def __init__(self):
        self.current_state = chess.Board()
        self.current_state.set_pieces()
        self.game_outcome = None  # 0 - player 1 won, 'draw' - draw, 1 - player 2 won, None - game not over

        self.resources = {}  # sprites for pygame

    def reinitialize(self):
        self.current_state = chess.Board()
        self.current_state.set_pieces()  # set up initial piece configuration
        self.current_player = 0
        self.game_outcome = None

    def clone(self):
        new_state = copy.copy(self)
        new_state.current_state = copy.copy(self.current_state)
        return new_state

    def set(self, state):
        self.current_state = state.current_state
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

    def take_action(self, action):
        reward = self.current_state.move_piece(action)

        self.current_state.last_action, previous_player = action, self.current_player
        self.update_current_player()

        self.current_state.cached_actions = []
        self.get_actions()
        if len(self.current_state.cached_actions) == 0:
            self.game_outcome = previous_player if self.current_state.is_checked(self.get_current_color()) else 'draw'
            if self.game_outcome != 'draw':
                reward = self.current_state.piece_values['k']

        # The current player gets the opposite of the reward (e.g. losing a piece)
        return np.array([-1 * reward if player_idx == self.current_player else reward
                         for player_idx in range(self.num_players)])

    def get_actions(self):
        if len(self.current_state.cached_actions) == 0:
            self.current_state.cached_actions = self.current_state.get_actions(self.get_current_color())
        return self.current_state.cached_actions

    def get_current_color(self):
        return 'white' if self.current_player == 0 else 'black'

    def get_value_bounds(self):
        king_value = self.current_state.piece_values['k']  # defeat / victory
        queen_value = self.current_state.piece_values['q']
        return {'defeat': -1 * king_value, 'victory': king_value,
                'min non-terminal': -1 * queen_value, 'max non-terminal': queen_value,
                'pre-computed min': None, 'pre-computed max': None,
                'evaluation function': None}

    def is_terminal(self):
        return self.game_outcome is not None

    def render(self):
        """Render the game board, creating a tkinter window if needed."""
        if not hasattr(self, 'screen'):
            pygame.init()
            self.width, self.height = 360, 360
            self.tile_size = int(self.width / self.current_state.width)  # assume width == height
            self.screen = pygame.display.set_mode((self.width, self.height))

            pygame.display.set_caption(self.env_name)
            if len(self.resources) == 0:
                self.load_resources("..\\dealers\\simulators\\chesscode\\sprites")
            icon = pygame.transform.scale(self.resources['kwhite'], (32, 32))
            pygame.display.set_icon(icon)

        pygame.event.clear()  # allows for pausing and debugging without losing rendering capability

        self.screen.blit(self.resources['background'], self.resources['background'].get_rect())
        for color in ('white', 'black'):  # recreate piece sets
            for piece in self.current_state.pieces[color]:
                # Load the image, scale it, and put it on the correct tile
                name = piece.abbreviation + piece.color
                image = self.resources[name]

                piece_rect = image.get_rect()
                piece_rect.move_ip(self.tile_size * piece.position[1],
                                   self.tile_size * piece.position[0])  # move in-place

                # Draw the piece
                self.screen.blit(image, piece_rect)

        pygame.display.update()  # update visible display

    def load_resources(self, path):
        """Load the requisite images for chess rendering from the given path."""
        image = pygame.image.load_basic(os.path.join(path, "board.bmp"))
        self.resources['background'] = pygame.transform.scale(image, (self.width, self.height))

        for abbreviation in self.current_state.piece_values.keys():  # load each piece
            for color in ('white', 'black'):  # load each color variant
                name = abbreviation + color  # construct image name
                image = pygame.image.load_extended(os.path.join(path, name + '.png'))
                self.resources[name] = pygame.transform.scale(image, (self.tile_size, self.tile_size))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return self.current_state.__str__()  # print board
