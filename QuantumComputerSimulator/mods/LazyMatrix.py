from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import copy
import numpy as np

class LazyMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        if Type == 'I':
            self.matrix = [lambda x: x[0], lambda x: x[1]]
        if Type == 'H':
            self.matrix = [lambda x: (x[0] + x[1]) / np.sqrt(2), lambda x: (x[0] - x[1]) / np.sqrt(2)]
        if Type == 'P':
            self.matrix = [lambda x: x[0], lambda x: np.exp(1j * args[0]) * x[1]]

        if Type == 'X':
            self.matrix = [lambda x: x[1], lambda x: x[0]]
        if Type == 'Y':
            self.matrix = [lambda x: 1j * x[1], lambda x: -1j * x[0]]
        if Type == 'Z':
            self.matrix = [lambda x: x[0], lambda x: -1 * x[1]]

        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0]

        if Type == 'CNOT':
            self.matrix = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':
            self.matrix = self.cv(args[0], args[1], args[2])
        if Type == 'CZ':
            self.matrix = self.cz(args[0], args[1], args[2])

        if Type == 'M0':
            self.matrix = [lambda x: x[0], lambda x: 0]
        if Type == 'M1':
            self.matrix = [lambda x: 0, lambda x: x[1]]

        self.dim = len(self.matrix)

    @classmethod
    def tensor_prod(cls, m1, m2):
        tp = []
        for i in range(0, m1.dim):
            for j in range(0, m2.dim):
                tp.append(lambda x, y=i, z=j: m1.matrix[y](
                    [m2.matrix[z]([x[m2.dim * k + l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('TP', tp)
        return new_matrix

    @classmethod
    def matrix_multiply(cls, m1, m2):
        mm = []
        for i in range(0, m1.dim):
            mm.append(
                lambda x, y=i: m1.matrix[y]([m2.matrix[k]([x[l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('MM', mm)
        return new_matrix

    @classmethod
    def inner_prod(cls, M):
        pass

    @classmethod
    def trace(cls, M):
        pass

    def cnot(self, d, c, t):
        digits = copy.deepcopy(d)
        cn = []

        index = super().CNOT_logic(digits, c, t)

        for i in range(0, len(index)):
            cn.append(lambda x, y=i: x[index[y]])

        return cn

    def cv(self, d, c, t):
        digits = copy.deepcopy(d)
        cv = []

        index = super().CV_logic(digits, c, t)

        for i in range(0, len(digits)):
            if index[i] == 1:
                cv.append(lambda x, y=i: 1j * x[y])
            else:
                cv.append(lambda x, y=i: x[y])

        return cv

    def cz(self, d, c, t):
        pass

    def output(self, inputs):
        # add conversion for vector input
        out = []
        for i in range(0, self.dim):
            out.append(self.matrix[i](inputs))

        return out


class LazyMatrixSingle(MatrixFrame):
    def __init__(self, Type, *args):
        pass
