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

        # set up quantum register
        self.Q_Register()

        # single input gates
        self.I = self.Matrix('I')
        self.H = self.Matrix('H')
        self.P = lambda theta: self.Matrix('P', theta)
        # How does single_gate deal with the extra argument required for P?

        # measuring gates
        self.M0 = self.Matrix('M0')
        self.M1 = self.Matrix('M1')

        # produce binary digits for 2 input gate logic
        self.binary = self.produce_digits()

        # gate inputs
        self.single_inputs = ["H", "P", "M0", "M1"]
        self.matrices = [self.H, self.P, self.M0, self.M1]

        self.double_inputs = ["CV", "CNOT", "CZ"]

        # tests
        print("qubit 0 in state 0")
        self.measure(0, 0)
        print("qubit 0 in state 1")
        self.measure(0, 1)
        print("qubit 1 in state 0")
        self.measure(1, 0)
        print("qubit 1 in state 1")
        self.measure(1, 1)

        inputs = [(["H"], [[0]]), (["CNOT"], [[0, 1]])]
        circ = self.gate_logic(inputs)
        self.psi = circ.output(self.psi)
        # Make gate_logic produce objects

        print("\n")
        print("qubit 0 in state 0")
        self.measure(0, 0)
        print("qubit 0 in state 1")
        self.measure(0, 1)
        print("qubit 1 in state 0")
        self.measure(1, 0)
        print("qubit 1 in state 1")
        self.measure(1, 1)

        # self.cv = self.Matrix('CV',self.binary,1,0)
        # print(self.cv.output([1,2,3,5]))

    def Q_Register(self):
        coeffs = []

        for i in range(0, self.N):
            alpha = np.random.random() + np.random.random() * 1j
            beta = np.random.random() + np.random.random() * 1j
            normF = np.sqrt(alpha * np.conj(alpha) + beta * np.conj(beta))

            alpha /= normF
            beta /= normF

            coeffs.append(np.array([[alpha], [beta]]))

        self.psi = coeffs[0]
        for i in range(1, self.N):
            self.psi = DenseMatrix.tensor_prod(self.psi, coeffs[i])

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

    def single_gates(self, gate, qnum):
        M = [0] * self.N

        for i in range(0, len(self.single_inputs)):
            for j in range(0, len(gate)):
                if self.single_inputs[i] == gate[j]:
                    for k in range(0, len(qnum[j])):
                        if gate[j] == "P":
                            pass
                        else:
                            M[qnum[j][k]] = self.matrices[i]

        for i in range(0, len(M)):
            if type(M[i]) != np.ndarray and type(M[i]) != self.Matrix:
                M[i] = self.I

        m = M[0]
        for i in range(1, len(M)):
            m = self.Matrix.tensor_prod(m, M[i])

        return m

    def double_gates(self, gate, qnum):

        if gate[0] == "CV":
            return self.Matrix("CV", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.Matrix("CNOT", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CZ":
            return self.Matrix("CZ", self.binary, qnum[0][0], qnum[0][1])

    def gate_logic(self, inputs):
        step_n = len(inputs)

        M = []

        for i in range(0, step_n):
            for j in range(0, len(self.single_inputs)):
                if self.single_inputs[j] in inputs[i][0]:
                    M.append(self.single_gates(inputs[i][0], inputs[i][1]))
            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    M.append(self.double_gates(inputs[i][0], inputs[i][1]))

        m = M[0]
        for i in range(1, len(M)):
            m = self.Matrix.matrix_multiply(m, M[i])

        return m

    def measure(self, qnum, state):
        inner_register = self.Matrix.inner_prod(self.psi)

        if state == 0:
            matrix = self.single_gates(["M0"], [[qnum]])
        elif state == 1:
            matrix = self.single_gates(["M1"], [[qnum]])

        QP = self.Matrix.trace(self.Matrix.matrix_multiply(matrix, inner_register))
        print(QP)

        x = []
        for i in range(0, 1000):
            if np.random.random() < QP:
                x.append(state)
            else:
                x.append(1 - state)

        # plt.hist(x)
        # plt.show()


class MatrixFrame(ABC):

    def __init__(self):
        pass

    def recog_digits(self, digits):
        N = int(np.log(len(digits)) / np.log(2))
        numbers = []
        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** (N - j - 1) * digits[i][j]
            numbers.append(num)
        return numbers

    def CNOT_logic(self, digits, c, t):
        N = int(np.log(len(digits)) / np.log(2))

        for i in range(0, 2 ** N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.recog_digits(digits)

        return index

    def CV_logic(self, digits, c, t):
        N = int(np.log(len(digits)) / np.log(2))
        index = []
        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                index.append(1)
            else:
                index.append(0)
        return index

    def CZ_logic(self, digits, c, t):
        return self.CV_logic(digits, c, t)

    @abstractmethod
    def tensor_prod(self, M1, M2):
        pass

    @abstractmethod
    def matrix_multiply(self, M1, M2):
        pass

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
            self.matrix = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':
            self.matrix = self.cv(args[0], args[1], args[2])
        if Type == 'CZ':
            self.matrix = self.cz(args[0], args[1], args[2])
        if Type == 'M0':
            self.matrix = np.array([[1, 0], [0, 0]])
        if Type == 'M1':
            self.matrix = np.array([[0, 0], [0, 1]])

    @classmethod
    def tensor_prod(cls, M1, M2):
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
        basis = self.Basis(N)

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

        for i in range(0, N):
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

        for i in range(0, N):
            if index[i] == 1:
                new_row = -1 * basis[i]
            else:
                new_row = basis[i]
            new_row.shape = (1, 2 ** N)
            cz.append(new_row)

        cz = np.asarray(np.matrix(np.asarray(cv)))

        return cz

    def output(self, inputs):
        return DenseMatrix.matrix_multiply(self.matrix, inputs)


class SparseMatrix(MatrixFrame):

    def __init__(self, Type: str, *args):
        if Type == 'I':  # identity gate
            self.matrix = np.array([0, 0, 1], [1, 1, 1])
        if Type == 'H':  # hadamard gate
            self.matrix = 1 / math.sqrt(2) * np.array([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, -1])
        if Type == 'TP' or Type == 'MM':  # tensor product or matrix multiplication
            self.matrix = args[0]  # 'matrix' to be first argument fed into the operation
        if Type == 'CNOT':
            pass
        if Type == 'CV':
            pass
        if Type == 'M0':
            self.matrix = np.array([0, 0, 1], [1, 1, 0])
        if Type == 'M1':
            self.matrix = np.array([1, 1, 1])
        self.dim = self.size_matrix()[0]

    def size_matrix(self):
        ncol = self.matrix[-1][0] + 1  # number of columns is the coloumn value of the last entry in the sparse matrix
        nr = 0  # number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(self.matrix)):
            if self.matrix[j][1] > nr:
                nr = self.matrix[j][1]
        nrow = nr + 1
        return (ncol, nrow)

    @classmethod
    def tensor_prod(cls, m1, m2):
        m2_col = self.size_matrix(m2)[0]  # STcol/SM1col = SM2col etc.
        m2_row = self.size_matrix(m2)[1]

        tensorprod = []
        for j in range(len(m1)):
            for i in range(len(m1)):
                column = m2_col * m1[j][0] + m2[i][0]
                row = m2_row * m1[j][1] + m2[i][1]
                value = m1[j][2] * m2[i][2]
                tensorprod.append([column, row, value])

        return tensorprod

    @classmethod
    def matrix_multiply(cls, m1, m2):
        # Convert SM1 and SM2 to a dictionaries with (row, col) keys and values for matrix manipulation when adding terms for matrix multiplication
        dict1 = {(row, col): val for row, col, val in m1}
        dict2 = {(row, col): val for row, c, v in m2}

        dict = {}
        for (r1, c1), v1 in dict1.items():  # iterate over SM1
            for (r2, c2), v2 in dict2.items():  # and SM2
                if c1 == r2:  # when the coloumn entry of SM1 and row entry of SM2 match, this is included in the non-zero terms for the matmul matrix
                    dict[(r1, c2)] = dict.get((r1, c2),
                                              0) + v1 * v2  # there may be more non-zero adding terms for each item in the matmul so the dictionary takes care of that

        matmul = [[r, c, v] for (r, c), v in dict.items()]  # return in sparse matric form
        return matmul

    def output(self, inputs):
        return matrix_multiply(self.matrix, inputs)


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


# computer
comp2 = QuantumComputer(2, 'Dense')

# comp3 = QuantumComputer(3,'Lazy')




