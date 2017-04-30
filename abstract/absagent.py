import abc
from abc import ABCMeta

class AbstractAgent:
	__metaclass__ = ABCMeta

	@abc.abstractmethod
	def select_action(self,state):
		raise NotImplementedError

	@abc.abstractmethod
	def get_agent_name(self):
		raise NotImplementedError