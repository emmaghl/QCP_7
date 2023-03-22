from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix


import copy
import numpy as np

class LazyMatrix(MatrixFrame):

    def __init__(self, Type, *args):
        '''
        Implements the Lazy method for quantum computing simulation.

        <b>Type</b> Gate to be built. <br>
        <b>*args</b> Position of control and target qubits.
        '''
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

        if Type == 'TP' or Type == 'MM' or Type == "General":
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

        if Type == 'zerocol':
            self.matrix = []
        if Type == 'onecol':
            self.matrix = []

        self.dim = len(self.matrix)

    @classmethod
    def quantum_register(cls, qnum):
        reg = []
        for i in range(0,qnum):
            reg.append([lambda x: x[i]])
        return LazyMatrix('General',reg)

    @classmethod
    def tensor_prod(cls, m2, m1):
        '''
        Lazy tensor product
        <b>m1</b> Gate 1
        <b>m2</b> Gate 2
        <b>return</b> Tensor product of Gate 1 with Gate 2
        '''
        tp = []
        for i in range(0, m1.dim):
            for j in range(0, m2.dim):
                tp.append(lambda x, y=i, z=j: m1.matrix[y](
                    [m2.matrix[z]([x[m2.dim * k + l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('TP', tp)
        return new_matrix

    @classmethod
    def matrix_multiply(cls, m1, m2):
        '''
        Use list comprehension to preform a matrix multiplication between two 'matrices'
        <b> m1</b> Gate 1
        <b> m2</b> Gate 2
        <b>return</b> Multiplication of gate 1 and gate 2
        '''
        mm = []
        for i in range(0, m1.dim):
            mm.append(
                lambda x, y=i: m1.matrix[y]([m2.matrix[k]([x[l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('MM', mm)
        return new_matrix

    @classmethod
    def inner_product(cls, M):
        return DenseMatrix.inner_product(M)

    @classmethod
    def trace(cls, M):
        return DenseMatrix.trace(M)

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
        digits = copy.deepcopy(d)
        cz = []

        index = super().CZ_logic(digits,c,t)

        for i in range(0,len(digits)):
            if index[i] == 1:
                cz.append(lambda x,y=i: -1*x[y])
            else:
                cz.append(lambda x,y=i: x[y])

        return cz

    def output(self,inputs):
        new_in = []
        for i in range(0,len(inputs)):
            new_in.append(inputs[i])
        out = []
        for i in range(0,self.dim):
            out.append(self.matrix[i](new_in))

        #To Vector form:
        out = np.array(out)
        out.shape = (len(out),1)
        return out