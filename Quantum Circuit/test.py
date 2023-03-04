from abc import ABC, abstractmethod


class Sparse(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __Make_Logic(self):
        '''Describe what it does'''
        pass

    def hello(self):
        pass


s = Sparse()
