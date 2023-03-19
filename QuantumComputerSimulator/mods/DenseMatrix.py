from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import numpy as np
import copy
import math

class DenseMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        '''
        Sets up gates in the dense method. All gates are full matrices and executed through standard linear algebra operations.
        <b>param Type<\b> Gate to be built
        <b>param args<\b>
        '''
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
        if Type == 'TP' or Type == 'MM' or Type == "general":
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

        if Type == 'zerocol':
            self.matrix = np.array([[1], [0]])
        if Type == 'onecol':
            self.matrix = np.array([[0], [1]])

    @classmethod
    def quantum_register(cls, qnum):
        register = np.array([[1, 0]])
        w = 2 ** (qnum) - 2
        for i in range(w):
            register = np.append(register, [0])
        register = np.array([register]).T
        return register

    @classmethod
    def tensor_prod(cls, M2, M1):
        '''
        Do the tensor product of matrix 1 and matrix 2.
        <b>param M2<\b> Matrix 2
        <b>param M1<\b> Matrix 1
        <b>return<\b> Tensor product of Matrix 1 with Matrix 2
        '''
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
        '''
        Multiply two matrices
        <b>param M1<\b> Matrix 1
        <b>param M2<\b> Matrix 2
        <b>return<\b> Matrix 1 multiplied by matrix 2
        '''
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
    def inner_product(cls, M):
        '''
        Find the inner product
        <b>param M<\b> input matrix
        <b>return<\b> inner product of state
        '''
        if type(M) == DenseMatrix:
            m = M.matrix
        else:
            m = M

        return DenseMatrix.matrix_multiply(m, np.transpose(np.conj(m)))

    @classmethod
    def conjugate(cls, M):
        return np.conj(M)
    @classmethod
    def trace(cls, M):
        '''
        Matrix trace
        <b>param M<\b> Input matrix
        <b>return<\b> matrix trace
        '''
        return np.trace(M.matrix)

    @classmethod
    def transpose(cls, M):
        if type(M) == DenseMatrix:
            m = M.matrix
        else:
            m = M
        return m.T

    def Basis(self, N):
        '''
        Define the basis state of N qubits
        <b>param N<\b> Number of Qubits
        <b>return<\b> basis state
        '''
        Q = []
        for i in range(0, 2 ** N):
            Q.append(np.zeros(2 ** N))
            Q[i][i] = 1
            Q[i].shape = (2 ** N, 1)
        return Q

    def cnot(self, d, c, t):
        '''
        Produce the multi-input gate CNOT, inherits from MatrixFrame and builds in dense method.
        '''
        digits = copy.deepcopy(d)
        cn = []

        index = super().CNOT_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2 ** N):
            new_row = basis[index[i]]
            new_row.shape = (1, 2 ** N)
            cn.append(new_row)

        cn = np.asarray(np.asmatrix(np.asarray(cn))) # need to see if can neaten this up
        return cn

    def cv(self, d, c, t):
        '''
        Build the control V gate, inherits from MatrixFrame and builds in dense method
        '''
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

        cv = np.asarray(np.asmatrix(np.asarray(cv)))

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

        cz = np.asarray(np.asmatrix(np.asarray(cz)))

        return cz

    def CROT_k(self, k):
        UROT_k =  np.array([[1, 0], [0, math.exp(2*math.pi*1j/2**k)]])
        CROT_k = cls.tensor_prod(self.M0, self.I) + cls.tensor_prod(self.M1, UROT_k)
        return CROT_k

    def output(self, inputs):
        '''
        Output of the DenseMatrix class, returns this when called by Quantum Computer
        '''
        return DenseMatrix.matrix_multiply(self.matrix, inputs)

    def apply_register(self, input_vector: list) -> list:
        '''Returns the output state vector.'''
        amplitudes = self.output([[v] for v in input_vector])
        return [amp[0]*np.conjugate(amp)[0] for amp in amplitudes.matrix]