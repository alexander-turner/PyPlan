import copy
import os
import pygame
from abstract import abstract_state
from dealers.simulators.chesscode import chess


class ChessState(abstract_state.AbstractState):
    env_name = "Chess"
    num_players = 2

    def __init__(self):
        self.current_state = chess.Board()
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 'draw' - draw, 1 - player2 is winner, None - game not over

        self.resources = {}  # image resources for pygame

    def reinitialize(self):
        self.current_state = chess.Board()
        self.current_player = 0
        self.game_outcome = None

        self.resources = {}

    def clone(self):
        new_state = copy.copy(self)
        new_state.current_state = copy.deepcopy(self.current_state)
        return new_state

    def set(self, state):
        self.current_state = copy.deepcopy(state.current_state)
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome
        
    def take_action(self, action):  # todo negative reward if piece lost?
        reward = self.current_state.update_board(action)
        action.has_moved = True  # mark that the piece has been moved (for castling purposes)

        previous_player, self.current_player = self.current_player, (self.current_player + 1) % self.num_players

        actions = self.get_actions()
        if len(actions) == 0:
            self.game_outcome = previous_player if self.current_state.is_legal(self.get_current_color()) else 'draw'
            if self.game_outcome != 'draw':  # todo how do we handle draws / rewards?
                reward = self.current_state.piece_values['k']
        # todo check other player too?

        rewards = [reward for _ in range(self.num_players)]
        rewards[self.current_player] *= -1  # the current player gets the opposite of the reward (e.g. losing a piece)
        return rewards

    def get_actions(self):
        return self.current_state.players[self.get_current_color()].get_actions()

    def number_of_players(self):
        return self.num_players

    def get_current_player(self):
        return self.current_player

    def get_current_color(self):
        return 'white' if self.current_player == 0 else 'black'

    def set_current_player(self, player_index):
        self.current_player = player_index

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
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.load_resources("..\\dealers\\simulators\\chesscode\\sprites")

        tile_size = int(self.width / self.current_state.width)  # assume width == height

        self.screen.blit(self.resources['background'], self.resources['background'].get_rect())

        for piece in self.current_state.players['white'].pieces + self.current_state.players['black'].pieces:
            # Load the image, scale it, and put it on the correct tile
            name = piece.abbreviation + piece.color
            image = self.resources[name]

            piece_rect = image.get_rect()
            piece_rect.move_ip(tile_size * piece.position[1], tile_size * piece.position[0])  # move in-place

            # Draw the piece
            self.screen.blit(image, piece_rect)

        pygame.display.flip()  # update visible display
        if self.is_terminal():
            pygame.quit()

    def load_resources(self, path):
        """Load the requisite images for chess rendering from the given path."""
        tile_size = int(self.width / self.current_state.width)  # assume width == height

        image = pygame.image.load_basic(os.path.join(path, "board.bmp"))
        self.resources['background'] = pygame.transform.scale(image, (self.width, self.height))

        for abbreviation in self.current_state.piece_values.keys():  # load each piece
            for color in ('white', 'black'):  # load each color variant
                name = abbreviation + color  # construct image name
                image = pygame.image.load_extended(os.path.join(path, name + '.png'))
                self.resources[name] = pygame.transform.scale(image, (tile_size, tile_size))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        return self.current_state.__str__()  # print board
