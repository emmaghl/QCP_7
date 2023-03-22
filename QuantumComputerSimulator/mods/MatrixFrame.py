import numpy as np
from abc import ABC, abstractmethod

class MatrixFrame(ABC):

    def __init__(self):
        '''
        Abstract class - forces implementations of necessary functions for Dense/Lazy/Sparse methods.
        '''
        self.matrix = 0
        pass

    def __recog_digits(self, digits):
        '''
        Used for creating the index for CNOT gate.

        <b>digits</b> The binary basis.
        '''
        N = int(np.log(len(digits)) / np.log(2))
        numbers = []
        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** j * digits[i][j]
            numbers.append(num)
        return numbers

    def CNOT_logic(self, digits_in, c, t):
        '''
        Logic to build the Control Not gate, has children in dense and sparse

        <b>digits_in</b> The binary basis. <br>
        <b>c</b> Control qubit. <br>
        <b>t</b> Target Qubit.<br>
        <b>return</b> The index to place that will make gate.
        '''
        N = int(np.log(len(digits_in)) / np.log(2))

        digits_out = digits_in
        for i in range(0, 2 ** N):
            if digits_in[i][c] == 1:
                digits_out[i][t] = 1 - digits_out[i][t] % 2

        index = self.__recog_digits(digits_out)

        return index

    def CV_logic(self, digits, c, t):
        '''
        Logic to build a Control V gate, has children in DenseMatrix and SparseMatrix.

        <b>digits_in</b> The binary basis. <br>
        <b>c</b> Control qubit. <br>
        <b>t</b> Target Qubit.<br>
        <b>return</b> The index to place that will make gate.
        '''
        N = int(np.log(len(digits)) / np.log(2))
        index = []
        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                index.append(1)
            else:
                index.append(0)
        return index

    def CZ_logic(self, digits, c, t):
        '''
        CZ logic gate, same as CV.

        <b>digits_in</b> The binary basis. <br>
        <b>c</b> Control qubit. <br>
        <b>t</b> Target Qubit.<br>
        <b>return</b> The index to place that will make gate.
        '''

        return self.CV_logic(digits, c, t)

    def apply_register(self, input_vector: list) -> list:
        '''Returns the output state vector.'''
        amplitudes = self.output(input_vector)
        return [amp[0]*np.conjugate(amp)[0] for amp in amplitudes]

    @abstractmethod
    def tensor_prod(self, M1, M2):
        '''
        Tensor product of matrix 1 and matrix 2.

        <b>M2</b> Matrix 2 </br>
        <b>M1</b> Matrix 1 </br>
        <b>return</b> Tensor product of Matrix 1 with Matrix 2
        '''
        pass

    @abstractmethod
    def matrix_multiply(self, M1, M2):
        '''
        Multiply two matrices
        <b>M1</b> Matrix 1 </br>
        <b>M2</b> Matrix 2 </br>
        <b>return</b> Matrix 1 multiplied by matrix 2
        '''
        pass

    @abstractmethod
    def inner_product(self, M):
        '''
        Find the inner product

        <b>M</b> input matrix </br>
        <b>return</b> inner product of state
        '''
        pass

    @abstractmethod
    def trace(self, M):
        '''
        Find the trace of matrix object

        <b>M</b> Matrix object.
        '''
        pass

    @abstractmethod
    def output(self, input):
        '''Returns the amplitudes of the state vector that is produced from applying the register.

        <b>input</b> The register.
        '''
        pass

