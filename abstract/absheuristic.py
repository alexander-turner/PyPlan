import abc
from abc import ABCMeta

class AbstractHeuristic:
	__metaclass__ = ABCMeta

	@abc.abstractmethod
	def get_heuristic_name(self):
		raise NotImplementedError

	@abc.abstractmethod
	def evaluate(self, state):
		raise NotImplementedError