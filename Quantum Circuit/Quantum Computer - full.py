# Quantum Computer - complete version

import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Interface(ABC):
    pass


# NOTE: watch for user errors!

class QuantumComputer(Interface):

    def __init__(self, qubits, matrix_type="Dense"):
        self.N = qubits

        if matrix_type == "Dense":
            self.Matrix = DenseMatrix
        if matrix_type == "Sparse":
            self.Matrix = SparseMatrix
        if matrix_type == "Lazy":
            self.Matrix = LazyMatrix

        # set up basis and quantum register
        #
        #

        # single input gates
        self.I = self.Matrix('I')
        self.H = self.Matrix('H')
        self.P = lambda theta: self.Matrix('P', theta)

        # measuring gates
        self.M0 = self.Matrix('M0')
        self.M1 = self.Matrix('M1')

        # produce binary digits for 2 input gate logic
        self.binary = self.produce_digits()

        # gate inputs
        self.single_inputs = ["H", "P", "M0", "M1"]
        self.matrices = [self.H, self.P, self.M0, self.M1]

        self.double_inputs = ["CV", "CNOT"]

        # self.cnot = self.Matrix('CNOT',self.N,1,0)
        # print(self.cnot.output([1,2,3,5]))


class MatrixFrame(object):

    def __init__(self):
        self.binary = self.produce_digits(

    def produce_digits(self):
        digits = []
        for i in range(0, 2 ** self.N):
            digit = []
            if i < (2 ** self.N) / 2:
                digit.append(0)
            else:
                digit.append(1)
            for j in range(1, self.N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** self.N) / (2 ** (k + 1))
                if x < (2 ** self.N) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        return digits

    def recog_digits(self, digits):
        numbers = []
        for i in range(0, 2 ** self.N):
            num = 0
            for j in range(0, self.N):
                num += 2 ** (self.N - j - 1) * digits[i][j]
            numbers.append(num)
        return numbers

    def CNOT_logic(self, c, t):
        digits = copy.deepcopy(self.binary)

        for i in range(0, 2 ** self.N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.recog_digits(self.N, digits)

        return index


class DenseMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        if Type == 'H':
            self.matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        if Type == 'I':
            self.matrix = np.array([[1, 0], [0, 1]])
        if Type == 'P':
            self.matrix = np.array([[1, 0], [1, np.exp(1j * args[0])]])
        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0]
        if Type == 'CNOT':
            pass
        if Type == 'CV':
            pass
        if Type == 'M0':
            self.matrix = np.array([[1, 0], [0, 0]])
        if Type == 'M1':
            self.matrix = np.array([[0, 0], [0, 1]])

    @classmethod
    def tensor_prod(cls, m1, m2):
        R = []
        if len(m1.shape) > 1:
            for i in range(len(m1)):
                R.append(m1[i][0] * m2)
                for j in range(1, len(m1[i])):
                    R[i] = np.concatenate((R[i], (m1[i][j] * m2)), axis=1)
            C = R[0]
            for i in range(1, len(m1)):
                C = np.concatenate((C, R[i]), axis=0)
        else:
            for i in range(len(m1)):
                R.append(m1[i] * m2)
            C = R[0]
            if m1.shape[0] > 0:
                ax = 0
            else:
                ax = 1
            for i in range(1, len(m1)):
                C = np.concatenate((C, R[i]), axis=ax)

        return C

    @classmethod
    def matrix_multiply(cls, m1, m2):
        M = np.zeros(len(m1) * len(m2[0]), dtype='complex')
        M.shape = (len(m1), len(m2[0]))

        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m2)):
                    M[i][j] += m1[i][k] * m2[k][j]

        return M


class SparseMatrix(MatrixFrame):
    def __init__(self):
        pass


class LazyMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        if Type == 'I':
            self.matrix = [lambda x: x[0], lambda x: x[1]]
        if Type == 'H':
            self.matrix = [lambda x: (x[0] + x[1]) / np.sqrt(2), lambda x: (x[0] - x[1]) / np.sqrt(2)]
        if Type == 'P':
            self.matrix = [lambda x: x[0], lambda x: np.exp(1j * args[0]) * x[1]]
        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0]
        if Type == 'CNOT':
            self.matrix = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':
            self.matrix = self.cv(args[0], args[1], args[2])
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

    def cnot(self, n, c, t):
        cn = []

        index = QuantumComputer.CNOT_logic(c, t)

        for i in range(0, len(index)):
            cn.append(lambda x, y=i: x[index[y]])

        return cn

    def cv(self, n, c, t):
        cv = []

        digits = QuantumComputer.produce_digits(n)

        for i in range(0, 2 ** n):
            if digits[i][c] == 1 and digits[i][t] == 1:
                cv.append(lambda x, y=i: 1j * x[y])
            else:
                cv.append(lambda x, y=i: x[y])

        return cv

    def output(self, inputs):
        out = []
        for i in range(0, self.dim):
            out.append(self.matrix[i](inputs))

        return out


class LazyMatrixSingle(MatrixFrame):
    def __init__(self, Type, *args):
        pass


# computer
# comp2 = QuantumComputer(2,'Dense')

comp3 = QuantumComputer(3, 'Lazy')
