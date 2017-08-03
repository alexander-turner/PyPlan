# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"
    name = "Left Turn Agent"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH

        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP

    def select_action(self, state):
        """For use with pacman (simulator)."""
        legal = state.get_actions()
        current = state.current_state.data.agentStates[0].configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH

        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    name = "Greedy Agent"

    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction is not None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        scores = []
        for action in legal:
            sim_state = state
            new_state = sim_state.generateSuccessor(0, action)
            scores.append((self.evaluationFunction(new_state), action))
        bestScore = max(scores)[0]
        bestActions = [pair[1] for pair in scores if pair[0] == bestScore]
        return random.choice(bestActions)

    def select_action(self, state):
        """For use with pacman (simulator)."""
        # Generate candidate actions
        legal = state.get_actions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        scores = []
        for action in legal:
            sim_state = state.current_state
            new_state = sim_state.generateSuccessor(0, action)
            scores.append((self.evaluationFunction(new_state), action))
        bestScore = max(scores)[0]
        bestActions = [pair[1] for pair in scores if pair[0] == bestScore]
        return random.choice(bestActions)


def scoreEvaluation(state):
    return state.getScore()
