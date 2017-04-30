import abc
from abc import ABCMeta

class AbstractBanditAlg:
	__metaclass__ = ABCMeta

	@abc.abstractmethod
	def get_bandit_name(self):
		raise NotImplementedError

	@abc.abstractmethod
	def initialize(self):
		raise NotImplementedError

	@abc.abstractmethod
	def update(self, arm, reward):
		raise NotImplementedError

	@abc.abstractmethod
	def select_best_arm(self):
		raise NotImplementedError

	@abc.abstractmethod
	def select_pull_arm(self):
		raise NotImplementedError

	@abc.abstractmethod
	def get_best_reward(self):
		raise NotImplementedError

	@abc.abstractmethod
	def get_num_pulls(self, arm):
		raise NotImplementedError

	@abc.abstractmethod
	def select_action(self, state):
		raise NotImplementedError

	@abc.abstractmethod
	def get_agent_name(self):
		raise NotImplementedError