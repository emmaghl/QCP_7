from QuantumComputerSimulator.mods.PrintingCircuit import PrintingCircuit
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix
from QuantumComputerSimulator.mods.SparseMatrix import SparseMatrix
from QuantumComputerSimulator.mods.LazyMatrix import LazyMatrix
from QuantumComputerSimulator.check import check

import numpy as np
from abc import ABC

class Interface(ABC):
    pass

class QuantumComputer(Interface):

    def __init__(self, qubits: int, matrix_type:str = "Dense"):
        '''Simulates quantum circuits via Dense, Sparse and Lazy methods.'''
        self.N = qubits

        check.check_in_list(matrix_type, ["Dense", "Sparse", "Lazy"])

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
        self.X = self.Matrix('X')
        self.Y = self.Matrix('Y')
        self.Z = self.Matrix('Z')
        # How does single_gate deal with the extra argument required for P?

        # measuring gates
        self.M0 = self.Matrix('M0')
        self.M1 = self.Matrix('M1')

        # produce binary digits for 2 input gate logic
        self.binary = self.produce_digits()

        # gate inputs
        self.single_inputs = ["H", "P", "X", "Y", "Z", "M0", "M1"]
        self.matrices = [self.H, self.P, self.X, self.Y, self.Z, self.M0, self.M1]

        self.double_inputs = ["CV", "CNOT", "CZ"]

        self.__gate_history = []

        # tests
        '''
        print("qubit 0 in state 0")
        self.measure(0,0)
        print("qubit 0 in state 1")
        self.measure(0,1)
        print("qubit 1 in state 0")
        self.measure(1,0)
        print("qubit 1 in state 1")
        self.measure(1,1)

        inputs = [(["H"],[[0]]),(["CNOT"],[[0,1]])]
        circ = self.gate_logic(inputs)
        self.psi = circ.output(self.psi)
        #Make gate_logic produce objects

        print("\n")
        print("qubit 0 in state 0")
        self.measure(0,0)
        print("qubit 0 in state 1")
        self.measure(0,1)
        print("qubit 1 in state 0")
        self.measure(1,0)
        print("qubit 1 in state 1")
        self.measure(1,1)
        '''

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

    def print_circuit(self):
        '''
        WARNING: Need to call `print_circuit_ascii` from terminal/cmd and will clear the terminal screen.
            Prints the quantum circuit in an ascii format on the terminal.
        '''
        pc = PrintingCircuit(self.__gate_history, self.N)
        pc.print_circuit_ascii()

    def produce_digits(self):  # this is the flipped basis, working on
        digits = []
        for i in range(0, 2 ** self.N):
            digit = []
            if i < (2 ** self.N) / 2:
                digit.append(0)
            else:
                digit.insert(0, 1)
            for j in range(1, self.N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** self.N) / (2 ** (k + 1))
                if x < (2 ** self.N) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        digits = np.flip(digits, axis=1)
        return digits

    def produce_digits2(self):
        digits = []
        for i in range(0, 2 ** self.N):
            digit = []
            if i < (2 ** self.N) / 2:
                digit.append(0)
            else:
                digit.insert(0, 1)
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

    def gate_logic(self, inputs, add_gate_name: str = ""):

        self.__validate_gate_logic_inputs(inputs)

        step_n = len(inputs)

        # Add the gates to the gate history for printing later.
        if add_gate_name == "": # If not defining a custom name
            [self.__gate_history.append(i) for i in inputs]
        else: # If defining a gate with a custom Name
            self.__gate_history.append(([add_gate_name], [[0, self.N-1]]))

        M = []

        for i in range(0, step_n):
            for j in range(0, len(self.single_inputs)):
                if self.single_inputs[j] in inputs[i][0]:
                    M.append(self.single_gates(inputs[i][0], inputs[i][1]))
            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    M.append(self.double_gates(inputs[i][0], inputs[i][1]))

        M = np.flip(M, axis=0)
        m = M[0]
        for i in range(1, len(M)):
            m = self.Matrix.matrix_multiply(m, M[i])

        return m

    def measure_any(self, qnum, state):
        inner_register = self.Matrix.inner_prod(self.psi)

        if state == 0:
            matrix = self.single_gates(["M0"], [[qnum]])
        elif state == 1:
            matrix = self.single_gates(["M1"], [[qnum]])

        QP = self.Matrix.trace(self.Matrix.matrix_multiply(matrix, inner_register))

        x = []
        for i in range(0, 1000):
            if np.random.random() < QP:
                x.append(state)
            else:
                x.append(1 - state)

        # plt.hist(x)
        # plt.show()

    def get_probabilities(self, glued_circuit, input_vector = np.nan):
        temp_vec = np.zeros((2**self.N))
        temp_vec[0] = 1
        outVec = np.matmul(glued_circuit,temp_vec)
        props = {}
        for i, basis in enumerate(self.binary):
            string_basis = ''.join([str(j) for j in basis[::-1]])
            props[string_basis] = np.real(outVec[i]*np.conjugate(outVec[i]))

        return props

    def __validate_gate_logic_inputs(self, inputs):
        check.check_type(inputs, list)

        for time_step in inputs:
            check.check_type(time_step, tuple)
            check.check_type(time_step[0], list)
            check.check_type(time_step[1], list)
            for gate in time_step[0]: #Looping through gates, check to see they're recognisable.
                check.check_type(gate, str)
                check.check_in_list(gate, self.single_inputs + self.double_inputs)
            for gate_positions in time_step[1]:
                check.check_type(gate_positions, list)







# computer
'''comp2 = QuantumComputer(4, 'Dense')

input1 = [(["CZ"], [[1, 0]])]

circ = comp2.gate_logic(input1)'''

# print(circ.matrix)


'''
comp = QuantumComputer(3,'Dense')

input1 = [(["H"],[[0,1,2]])]
input2 = [(["H"],[[0]]),(["H"],[[1]]),(["H"],[[2]])]

input3 = [(["H","Y"],[[0,2],[1]])]
input4 = [(["H"],[[0]]),(["Y"],[[1]]),(["H"],[[2]])]

circ1 = comp.gate_logic(input1)
circ2 = comp.gate_logic(input2)

circ3 = comp.gate_logic(input3)
circ4 = comp.gate_logic(input4)

print(circ3.matrix)
print(circ4.matrix)

#print(circ1.matrix)
#print(circ2.matrix)

#new_psi1 = comp.output(circ1)
#new_psi2 = comp.output(circ2)

#comp3 = QuantumComputer(3,'Lazy')

'''




