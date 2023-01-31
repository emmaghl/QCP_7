import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.X = Gates.X()
        self.Y = Gates.Y()
        self.Z = Gates.Z()
        self.H = Gates.Hadamard()
        self.RN = Gates.Root_Not()

    def Register(self):
        def Tensor_Prod(self):
            pass


class Gates:
    def __init__(self):
        pass

    def X(self):
        return np.array([[0, 1], [1, 0]])

    def Y(self):
        return np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex)

    def Z(self):
        return np.array([[1, 0], [0, -1]])

    def Root_Not(self):
        return 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]])

    def Hadamard(self):
        return np.array([[1, 1], [1, -1]])

    def Phase(self):
        return np.array([[1, 0], [1, 0 + 1j]], dtype=complex)

