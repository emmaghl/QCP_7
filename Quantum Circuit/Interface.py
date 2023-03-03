
import numpy as np
from abc import ABC, abstractmethod


class QuantumComputer(ABC):
    @abstractmethod
    def __init__(self, numQ):
        self.N = numQ

    @abstractmethod
    def TensorProduct(self,v1,v2):
        pass

    @abstractmethod
    def MatrixMultiply(self,m1,m2):
        pass

class Dense(QuantumComputer):
    def TensorProduct(self,v1,v2):
        pass

    def MatrixMultiply(self,m1,m2):
        pass

class Sparse(QuantumComputer):
    def __init__(self):

class Lazy(QuantumComputer):
    def __init__(self):



