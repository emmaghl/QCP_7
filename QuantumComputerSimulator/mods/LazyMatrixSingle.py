from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import copy
import numpy as np


class LazyMatrixSingle(MatrixFrame):

    def __init__(self, Type, *args):
        '''
        Set up the lazy gates using lambda functions.
        :param Type: Gate to be initialised
        :param args:
        '''
        if Type == 'I':
            self.matrix = lambda x: [x[0], x[1]]
            self.dim = 2
        if Type == 'H':
            self.matrix = lambda x: [(x[0] + x[1]) / np.sqrt(2), (x[0] - x[1]) / np.sqrt(2)]
            self.dim = 2
        if Type == 'P':
            self.matrix = lambda x: [x[0], np.exp(1j * args[0]) * x[1]]
            self.dim = 2

        if Type == 'X':
            self.matrix = lambda x: [x[1], x[0]]
            self.dim = 2
        if Type == 'Y':
            self.matrix = lambda x: [1j * x[1], -1j * x[0]]
            self.dim = 2
        if Type == 'Z':
            self.matrix = lambda x: [x[0], -1 * x[1]]
            self.dim = 2

        if Type == 'TP' or Type == 'MM' or Type == "General":
            self.matrix = args[0]
            self.dim = args[1]

        if Type == 'CNOT':
            self.matrix, self.dim = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':
            self.matrix, self.dim = self.cv(args[0], args[1], args[2])
        if Type == 'CZ':
            self.matrix, self.dim = self.cz(args[0], args[1], args[2])

        if Type == 'M0':
            self.matrix = lambda x: [x[0], 0]
            self.dim = 2
        if Type == 'M1':
            self.matrix = lambda x: [0, x[1]]
            self.dim = 2

        if Type == 'zerocol':
            self.matrix = []
        if Type == 'onecol':
            self.matrix = []

    @classmethod
    def quantum_register(cls, qnum):
        pass

    @classmethod
    def tensor_prod(cls, m2, m1):
        '''
        Lazy tensor product
        :param m1: Gate 1
        :param m2: Gate 2
        :return: Tensor product of Gate 1 with Gate 2
        '''

        tp = lambda x: [[m1.matrix(
            [[m2.matrix([x[m2.dim * n + m] for m in range(0, m2.dim)]) for n in range(0, m1.dim)][i][j] for i in
             range(0, m1.dim)]) for j in range(0, m2.dim)][l][k] for k in range(0, m1.dim) for l in range(0, m2.dim)]

        return LazyMatrixSingle('TP', tp, m1.dim * m2.dim)

    @classmethod
    def matrix_multiply(cls, m1, m2):
        '''
        Lazy matrix multiplication between two 'matrices'
        :param m1: Gate 1
        :param m2: Gate 2
        :return: Multiplication of gate 1 and gate 2
        '''
        mm = lambda x: m1.matrix(m2.matrix(x))

        return LazyMatrixSingle('MM', mm, m1.dim)

    @classmethod
    def inner_product(cls, M):
        pass

    @classmethod
    def trace(cls, M):
        pass

    @classmethod
    def conjugate(cls, M):
        pass

    def cnot(self, d, c, t):
        digits = copy.deepcopy(d)

        index = super().CNOT_logic(digits, c, t)

        return lambda x: [x[index[i]] for i in range(0, len(index))], int(np.log(len(index)) / np.log(2))

    def cv(self, d, c, t):
        digits = copy.deepcopy(d)
        cv = []

        index = super().CV_logic(digits, c, t)

        cv = lambda x: [1j * x[i] if index[i] == 1 else x[i] for i in range(0, len(index))]

        return cv, int(np.log(len(index)) / np.log(2))

    def cz(self, d, c, t):
        digits = copy.deepcopy(d)
        cz = []

        index = super().CZ_logic(digits, c, t)

        cz = lambda x: [-1 * x[i] if index[i] == 1 else x[i] for i in range(0, len(index))]

        return cz, int(np.log(len(index)) / np.log(2))

    def output(self, inputs):
        new_in = []
        for i in range(0, len(inputs)):
            new_in.append(inputs[i])

        out = self.matrix(new_in)

        # To Vector form:
        out = np.array(out)
        out.shape = (len(out), 1)
        return out

