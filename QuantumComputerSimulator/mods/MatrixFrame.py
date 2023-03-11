import numpy as np
from abc import ABC, abstractmethod

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

    def reverse_digits(self, d):
        temp = [[d[i*2], d[i*2+1]] for i in range(int(len(d)/2))]
        temp = np.flip(temp, axis=0)
        unpack = []
        for i in temp:
            for j in i:
                unpack.append(j)
        return unpack

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
