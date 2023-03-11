from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import numpy as np
import copy

class DenseMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        if Type == 'H':
            self.matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
        if Type == 'I':
            self.matrix = np.array([[1, 0], [0, 1]])
        if Type == 'P':
            self.matrix = np.array([[1, 0], [1, np.exp(1j * args[0])]])

        if Type == 'X':
            self.matrix = np.array([[0, 1], [1, 0]])
        if Type == 'Y':
            self.matrix = np.array([[0, 0 - 1j], [0 + 1j, 0]], dtype=complex)
        if Type == 'Z':
            self.matrix = np.array([[1, 0], [0, -1]])

        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0]

        if Type == 'CNOT':
            self.matrix = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':
            self.matrix = self.cv(args[0], args[1], args[2])
        if Type == 'CZ':
            self.matrix = self.cz(args[0], args[1], args[2])

        if Type == 'M0':
            self.matrix = np.array([[1, 0], [0, 0]])
        if Type == 'M1':
            self.matrix = np.array([[0, 0], [0, 1]])

    def produce_digits2(self):
        digits = []
        for i in range(0, 8):
            digit = []
            if i < (8) / 2:
                digit.append(0)
            else:
                digit.insert(0, 1)
            for j in range(1, 8):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (8) / (2 ** (k + 1))
                if x < (8) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        return digits

    @classmethod
    def tensor_prod(cls, M2, M1):
        if type(M1) == DenseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1
        if type(M2) == DenseMatrix:
            m2 = M2.matrix
        else:
            m2 = M2

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
        return DenseMatrix('TP', C)

    @classmethod
    def matrix_multiply(cls, M1, M2):
        if type(M1) == DenseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1
        if type(M2) == DenseMatrix:
            m2 = M2.matrix
        else:
            m2 = M2

        M = np.zeros(len(m1) * len(m2[0]), dtype='complex')
        M.shape = (len(m1), len(m2[0]))

        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m2)):
                    M[i][j] += m1[i][k] * m2[k][j]
        return DenseMatrix('MM', M)

    @classmethod
    def inner_prod(cls, M):
        return DenseMatrix.matrix_multiply(M.matrix, np.transpose(np.conj(M.matrix)))

    @classmethod
    def trace(cls, M):
        return np.trace(M.matrix)

    def Basis(self, N):
        Q = []
        for i in range(0, 2 ** N):
            Q.append(np.zeros(2 ** N))
            Q[i][i] = 1
            Q[i].shape = (2 ** N, 1)
        return Q

    def cnot(self, d, c, t):
        digits = copy.deepcopy(d)
        cn = []

        index = super().CNOT_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        # = np.asarray(self.produce_digits2()
        #print(basis)
        #basis = self.Basis(N)
        basis = self.Basis(N)
        #print(basis)

        for i in range(0, 2 ** N):
            new_row = basis[index[i]]
            new_row.shape = (1, 2 ** N)
            cn.append(new_row)

        cn = np.asarray(np.asmatrix(np.asarray(cn)))
        return cn

    def cv(self, d, c, t):
        digits = copy.deepcopy(d)
        cv = []

        index = super().CV_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2**N):
            if index[i] == 1:
                new_row = 1j * basis[i]
            else:
                new_row = basis[i]
            new_row.shape = (1, 2 ** N)
            cv.append(new_row)

        cv = np.asarray(np.matrix(np.asarray(cv)))

        return cv

    def cz(self, d, c, t):
        digits = copy.deepcopy(d)
        cz = []

        index = super().CZ_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2**N):
            if index[i] == 1:
                new_row = -1 * basis[i]
            else:
                new_row = basis[i]
            new_row.shape = (1, 2 ** N)
            cz.append(new_row)

        cz = np.asarray(np.matrix(np.asarray(cz)))

        return cz

    def output(self, inputs):
        return DenseMatrix.matrix_multiply(self.matrix, inputs)