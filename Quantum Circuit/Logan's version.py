# quantum computer simulation - Logan's Version
# current update as of 15/02

import numpy as np


class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits

        # computational bases
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state

        # produce basis and quantum register
        self.Basis()
        self.Q_Register()

        # single input gates
        self.I = np.array([[1, 0], [0, 1]])  # Identity gate

        self.H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # Hadamard: sends |0> to |+> and |1> to |->
        self.P = np.array([[1, 0], [1, 0 + 1j]])  # Phase shift: sends |0>+|1> to |0>+i|1> for a phase = pi/2
        # Note: improve phase shift gate for all phases
        # import other gates...

        self.single_inputs = ["H", "P"]
        self.matrices = [self.H, self.P]

        self.double_inputs = ["CV", "CNOT"]

        # produce binary digits for 2 input gate use
        self.binary = self.produce_digits()

    def Tensor_Prod(self, Q1, Q2):  # by emma
        R = []
        for i in range(len(Q1)):
            R.append(Q1[i][0] * Q2)
            for j in range(1, len(Q1[i])):
                R[i] = np.concatenate((R[i], (Q1[i][j] * Q2)), axis=1)

        C = R[0]
        for i in range(1, len(Q1)):
            C = np.concatenate((C, R[i]), axis=0)

        return C

    #if needed?
    def Mat_Mul(self, Q1, Q2):
        assert np.shape(Q1)[1] == np.shape(Q2)[0], "can't perform matrix multiplication"
        M = np.zeros(len(Q1) * len(Q2[0]))
        M.shape = (len(Q1), len(Q2[0]))
        print(M)

        for i in range(len(Q1)):  # rows of Q1
            for j in range(len(Q2[0])):  # columns of Q2
                for k in range(len(Q2)):  # rows of Q2
                    M[i][j] += Q1[i][k] * Q2[k][j]

        return M


    def Q_Register(self):
        j = 2 ** self.Register_Size
        coeffs = (0 + 0 * 1j) * np.zeros(j)
        for i in range(j):
            theta = np.random.random() * np.pi * 2
            coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j

        self.psi = coeffs
        self.psi.shape = (j, 1)

    def Basis(self):
        N = self.Register_Size
        self.Q = []

        for i in range(0, 2 ** N):
            self.Q.append(np.zeros(2 ** N))
            self.Q[i][i] = 1
            self.Q[i].shape = (2 ** N, 1)

    def Norm(self, array):
        total = 0
        for i in range(0, len(array)):
            total += np.sqrt(np.real(array[i]) ** 2 + np.imag(array[i]) ** 2)

        array /= total

        return array

    def CNOT(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        N = self.Register_Size

        cn = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.recog_digits(digits)

        for i in range(0, 2 ** N):
            new_row = self.Q[index[i]]
            new_row.shape = (1, 2 ** N)
            cn.append(new_row)

        cn = np.asarray(np.asmatrix(np.asarray(cn)))  # check for better method

        return cn

    def CV(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        N = self.Register_Size

        cv = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                new_row = 1j * self.Q[i]
            else:
                new_row = self.Q[i]
            new_row.shape = (1, 2 ** N)
            cv.append(new_row)

        cv = np.asarray(np.matrix(np.asarray(cv)))  # check for better method

        return cv

    def recog_digits(self, digits):
        N = self.Register_Size
        numbers = []

        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** (N - j - 1) * digits[i][j]
            numbers.append(num)

        return numbers

    def produce_digits(self):
        N = self.Register_Size
        digits = []
        for i in range(0, 2 ** N):
            digit = []
            if i < (2 ** N) / 2:
                digit.append(0)
            else:
                digit.append(1)
            for j in range(1, N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** N) / (2 ** (k + 1))
                if x < (2 ** N) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        return digits

    def Single_Gates(self, gate, qnum):
        M = [0] * self.Register_Size

        for i in range(0, len(self.single_inputs)):
            for j in range(0, len(gate)):
                if self.single_inputs[i] == gate[j]:
                    for k in range(0, len(qnum[j])):
                        M[qnum[j][k]] = self.matrices[i]

        for i in range(0, len(M)):
            if type(M[i]) != np.ndarray:
                M[i] = self.I

        m = M[0]
        for i in range(1, len(M)):
            m = self.Tensor_Prod(m, M[i])

        return m

    def Double_Gates(self, gate, qnum):

        if gate[0] == "CV":
            return self.CV(qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.CNOT(qnum[0][0], qnum[0][1])

    def Check_Inputs(self, gate, positions):  # by emma - MAKE THIS WORK WITH MY CODE

        assert len(gate) == len(
            positions), "unequal list lenghts"  # the number of gates should match the position lists

        # this is only one step. so only one logic gate can be applied to a single qubit. so must return an error
        # if any value within the position sub-lists is repeated
        list_check = []  # create a unique list with the each of the arguments of the sublists of positions
        for k in range(len(positions)):
            for i in range(len(positions[k])):
                list_check.append((positions[k])[i])

        assert len(list(list_check)) == len(
            set(list_check)), "repeated position value"  # ensure lenght of list is equal to lenght of unique values in list i.e. avoid repetiotion

    def Gate_Logic(self, inputs):
        N = self.Register_Size
        step_n = len(inputs)

        M = []

        for i in range(0, step_n):
            for j in range(0, len(self.single_inputs)):
                if self.single_inputs[j] in inputs[i][0]:
                    M.append(self.Single_Gates(inputs[i][0], inputs[i][1]))
            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    M.append(self.Double_Gates(inputs[i][0], inputs[i][1]))

        m = M[0]
        for i in range(1, len(M)):
            m = np.matmul(m, M[i])

        return m


class Interface(object):

    def __init__(self):
        pass


# main
comp = Quantum_Computer(2)


# CNOT gate made using only CV and Hadamard gates:
step1 = (["H"],[[1]])
step2 = (["CV"],[[0,1]])
step3 = (["CV"],[[0,1]])
step4 = (["H"],[[1]])

steps = [step1,step2,step3,step4]

mat = comp.Gate_Logic(steps)
mat = np.real(mat)
print(mat)

cnot = comp.CNOT(0,1)
print(cnot)