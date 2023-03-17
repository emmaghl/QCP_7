from QuantumComputerSimulator.mods.PrintingCircuit import PrintingCircuit
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix
from QuantumComputerSimulator.mods.SparseMatrix import SparseMatrix
from QuantumComputerSimulator.mods.LazyMatrix import LazyMatrix
from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame
from QuantumComputerSimulator.mods.check import check

import numpy as np
from abc import ABC

class Interface(ABC):
    pass

class QuantumComputer(Interface):

    def __init__(self, qubits: int, matrix_type:str = "Dense"):
        '''
        Simulates quantum circuits via Dense, Sparse and Lazy methods.
        <b>param: qubits<\b> Number of Qubits
        <b>param: matrix_type<\b> Desired type of matrix formulation to use. Default: Dense
        '''

        check.check_type(qubits, int)
        self.N = qubits

        check.check_type(matrix_type, str)
        check.check_in_list(matrix_type, ["Dense", "Sparse", "Lazy"])

        '''
        Take desired method: Dense, Sparse or Lazy; and set the quantum computer to use that method.
        '''
        if matrix_type == "Dense":
            self.Matrix = DenseMatrix
        if matrix_type == "Sparse":
            self.Matrix = SparseMatrix
        if matrix_type == "Lazy":
            self.Matrix = LazyMatrix

        '''
        Build the different gates using the desired method. Gate building is handled within the matrix method.
        '''
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

        #Initalise empty list to store history of gates used.
        self.__gate_history = []
        self.__custom_gate_names = []
        self.circuit = 0 # Will be a MatrixFrame object

    '''
    def Q_Register(self):
        '''
        Q_Register() build's the quantum register for the quantum computer for a given number of qubits.
        '''
        coeffs = []

        for i in range(0, self.N):
            alpha = np.random.random() + np.random.random() * 1j
            beta = np.random.random() + np.random.random() * 1j
            normF = np.sqrt(alpha * np.conj(alpha) + beta * np.conj(beta))

            alpha /= normF
            beta /= normF
    '''


    def print_circuit(self):
        '''
        Prints the circuit that is built from using `add_gate_to_circuit` function.

        WARNING: Need to call `print_circuit_ascii` from terminal/cmd and will clear the terminal screen.
            Prints the quantum circuit in an ascii format on the terminal.
        '''
        pc = PrintingCircuit(self.__gate_history, self.N, custom_gate_names=self.__custom_gate_names)
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

    def single_gates(self, gate, qnum):
        '''
        Build the single input gates to a 2^N matrix where N is the number of Qubits.
        <b>param: gate<\b> Specified gate to build
        <b>param: qnum<\b> Number of qubits
        <b>return<\b> Properly sized gate for number of qubits
        '''
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
        '''
        Use the Matrix method in the given desired method to build a multi-input gate.
        <b>param: gate<\b> Take specified multi-input gate
        <b>param: qnu<\b> number of qubits
        <b>return<\b> Returns correctly sized multi-input gate.
        '''
        if gate[0] == "CV":
            return self.Matrix("CV", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.Matrix("CNOT", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CZ":
            return self.Matrix("CZ", self.binary, qnum[0][0], qnum[0][1])

    def gate_logic(self, inputs, add_gate_name: str = "") -> np.array:
        '''
        Builds the quantum circuit of gates in time step of type tuple.
        <b>param: inputs<\b> Number of input gates?
        <b>param: add_gate_name<\b> Name of gates to be added.
        <b>return<\b> Circuit?
        '''

        check.check_type(add_gate_name, str)
        self.__validate_gate_logic_inputs(inputs)

        step_n = len(inputs)

        M = []

        for i in range(0, step_n):
            gate_type = []
            for j in range(0, len(self.single_inputs)):
                if self.single_inputs[j] in inputs[i][0]:
                    gate_type.append('single')

            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    gate_type.append('double')

            gate_single = []
            gate_double = []
            for j in range(0, len(gate_type)):
                if gate_type[j] == 'single':
                    gate_single.append(True)
                else:
                    gate_single.append(False)
                if gate_type[j] == 'double':
                    gate_double.append(True)
                else:
                    gate_double.append(False)

            if all(gate_single) == True:
                M.append(self.single_gates(inputs[i][0], inputs[i][1]))
            elif all(gate_double) == True:
                M.append(self.double_gates(inputs[i][0], inputs[i][1]))
            else:
                print(
                    "Input error: single gates and double gates must be in separate steps. Returning identity matrix instead.")
                return self.Matrix('I')

        M = np.flip(M, axis=0)
        m = M[0]
        for i in range(1, len(M)):
            m = self.Matrix.matrix_multiply(m, M[i])

        return m

    def measure_any(self, qnum, state, register):
        '''
        Generate the measurment of the quantum circuit. Once measured the system's wavefunction is collapsed.
        <b>param: qnum<\b> Number of qubits?
        <b>param: state<\b> State of the qubit
        '''
        inner_register = self.Matrix.inner_product(register)

        if state == 0:
            matrix = self.single_gates(["M0"], [[qnum]])
        elif state == 1:
            matrix = self.single_gates(["M1"], [[qnum]])

        QP = self.Matrix.trace(self.Matrix.matrix_multiply(matrix, inner_register))

        if (np.random.rand() < QP):
            result = 0
        else:
            result = 1
        return result

    def histogram(self):
        pass
        x = []
        for i in range(0, 1000):
            if np.random.random() < QP:
                x.append(state)
            else:
                x.append(1 - state)

        # plt.hist(x)
        # plt.show()

    def apply_register(self, inputs):
        outputs = self.circuit.output(inputs)
        probs = np.zeros(len(outputs))
        for i in range(len(outputs)):
            probs[i] = (self.Matrix.conjugate(outputs[i]))**2

        counts = 1000
        apply_dict = {inputs[i]: counts*probs[i] for i in inputs}
        return apply_dict

    def get_probabilities(self, glued_circuit: np.ndarray, input_vector: np.ndarray = np.nan):
        '''
        Generates the probability of a measured state?
        <b>param: glued_circuit<\b>
        <b>param: input_vector <\b>
        <b>return:<\b> Funciton returns probability
        '''
        num_qubits = 2**self.N
        check.check_type(glued_circuit, np.ndarray)
        check.check_array_shape(glued_circuit, (num_qubits, num_qubits))

        temp_vec = np.zeros((num_qubits))
        temp_vec[0] = 1
        if not np.isnan(input_vector):
            check.check_type(input_vector, np.ndarray)
            check.check_array_shape(input_vector, (num_qubits))
            temp_vec = input_vector

        outVec = np.matmul(glued_circuit,temp_vec)
        props = {}
        for i, basis in enumerate(self.binary):
            string_basis = ''.join([str(j) for j in basis[::-1]])
            props[string_basis] = np.real(outVec[i]*np.conjugate(outVec[i]))

        return props

    def add_gate_to_circuit(self, inputs: list, add_gate_name:str = ""):
        '''Adds the gates to the class ready to for building the circuit later.'''
        check.check_type(add_gate_name, str)
        self.__validate_gate_logic_inputs(inputs)

        if not add_gate_name == "": # If defining a gate with a custom Name
            length_of_gate_history = len(self.__gate_history)
            self.__custom_gate_names.append(
                [length_of_gate_history-1, length_of_gate_history + len(inputs), add_gate_name]
            )
        self.__gate_history = self.__gate_history + inputs

    def build_circuit(self) -> MatrixFrame:
        '''Builds the circuit from the gates added via `add_gate_to_circuit` function.'''
        self.circuit = self.gate_logic(self.__gate_history)
        return self.circuit

    def __validate_gate_logic_inputs(self, inputs: list):
        '''
        Check function. Checking gates make sense and are of the correct type.
        <b>param:<\b> inputs
        <b>return:<\b> Through other methods, pass/fail.
        '''
        check.check_type(inputs, list)

        for time_step in inputs:
            check.check_type(time_step, tuple)
            check.check_array_length(time_step, 2)
            check.check_type(time_step[0], list)
            check.check_type(time_step[1], list)
            check.check_array_length(time_step[0], len(time_step[1]))
            for gate in time_step[0]: #Looping through gates, check to see they're recognisable.
                check.check_type(gate, str)
                check.check_in_list(gate, self.single_inputs + self.double_inputs)
            for gate_positions in time_step[1]:
                check.check_type(gate_positions, list)
                for numbers in gate_positions:
                    check.check_type(numbers, int)




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




