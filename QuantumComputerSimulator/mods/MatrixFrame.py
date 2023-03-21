import numpy as np
from abc import ABC, abstractmethod

class MatrixFrame(ABC):

    def __init__(self):
        '''
        Abstract class
        '''
        self.matrix = 0
        pass

    def recog_digits(self, digits):
        '''

        <b>param digits<\b>
        '''
        N = int(np.log(len(digits)) / np.log(2))
        numbers = []
        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** j * digits[i][j]
            numbers.append(num)
        return numbers

    def reverse_digits(self, d):
        temp = [[d[i*2], d[i*2+1]] for i in range(int(len(d)/2))]
        temp = np.flip(temp, axis=0)
        unpack = []
        for i in temp:
            for j in i:
                unpack.append(j)
        return unpack

    def CNOT_logic(self, digits_in, c, t):
        '''
        Logic to build the Control Not gate, has children in dense and sparse
        <b>param digits_in<\b>
        <b>param c<\b>
        <b>param t<\b>
        '''
        N = int(np.log(len(digits_in)) / np.log(2))

        digits_out = digits_in
        for i in range(0, 2 ** N):
            if digits_in[i][c] == 1:
                digits_out[i][t] = 1 - digits_out[i][t] % 2

        index = self.recog_digits(digits_out)

        return index

    def CV_logic(self, digits, c, t):
        '''
        Logic to build a Control V gate, has children in DenseMatrix and SparseMatrix.
        <b>param digits<\b>
        <b>param c<\b>
        <b>param t<\b>
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
        return self.CV_logic(digits, c, t)

    def apply_register(self, input_vector: list) -> list:
        '''Returns the output state vector.'''
        amplitudes = self.output(input_vector)
        return [amp[0]*np.conjugate(amp)[0] for amp in amplitudes]

    @abstractmethod
    def tensor_prod(self, M1, M2):
        pass

    @abstractmethod
    def matrix_multiply(self, M1, M2):
        pass

    @abstractmethod
    def inner_product(self, M):
        pass

    @abstractmethod
    def trace(self, M):
        pass

    @abstractmethod
    def output(self, input):
        pass

