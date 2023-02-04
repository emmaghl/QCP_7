import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.X = Gates.X()
        self.Y = Gates.Y()
        self.Z = Gates.Z()
        self.H = Gates.Hadamard()
        self.CNot = Gates.C_not()

    def Register(self):
        def Tensor_Prod(self):
            pass


class Gates:
    def __init__(self, input):
        self.inpt = input

    def X(self):
        return np.matmul(np.array([[0, 1], [1, 0]]), self.inpt)

    def Y(self):
        return np.matmul(np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex),self.inpt)

    def Z(self):
        return np.matmul(np.array([[1, 0], [0, -1]]), self.inpt)

    def Root_Not(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]), self.inpt)

    def Hadamard(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), self.inpt)

    def Phase(self):
        return np.matmul(np.array([[1, 0], [1, 0 + 1j]], dtype=complex), self.inpt)

    def T(self):
        return np.matmul(np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex), self.inpt)

    def C_not(self):
        return np.matmul(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), self.inpt)

    def swap(self):
        return np.matmul(np.array([1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]), self.inpt)


