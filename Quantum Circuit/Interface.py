
import numpy as np
from abc import ABC, abstractmethod


class QuantumComputer(ABC):

    @abstractmethod
    def TensorProduct(self,v1,v2):
        pass

    @abstractmethod
    def MatrixMultiply(self,m1,m2):
        pass


class Dense(QuantumComputer,numQ):
    def __init__(self):
        self.N = numQ

    def TensorProduct(self,v1,v2):
        pass

    def MatrixMultiply(self,m1,m2):
        pass

class Sparse(QuantumComputer,numQ):
    def __init__(self):
        self.N = numQ

class Lazy(QuantumComputer,numQ):
    def __init__(self):
        self.N = numQ


