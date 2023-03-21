from QuantumComputerSimulator.mods.PrintingCircuit import PrintingCircuit
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix
from QuantumComputerSimulator.mods.SparseMatrix import SparseMatrix
from QuantumComputerSimulator.mods.LazyMatrix import LazyMatrix
from QuantumComputerSimulator.mods.LazyMatrixSingle import LazyMatrixSingle
from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame
from QuantumComputerSimulator.mods.check import check

import numpy as np
from abc import ABC
import random

class Interface(ABC):
    pass

class QuantumComputer(Interface):

    def __init__(self, qubits: int, matrix_type:str = "Dense"):
        '''
        Simulates quantum circuits via Dense, Sparse and Lazy methods.

        <b>qubits</b> Number of qubits in the circuit.<br>
        <b>matrix_type</b> Desired implementation of quantum computer to use. Default is Dense.
        '''

        check.check_type(qubits, int)
        self.N = qubits

        check.check_type(matrix_type, str)
        check.check_in_list(matrix_type, ["Dense", "Sparse", "Lazy", "LazySingle"])

        '''
        Take desired method: Dense, Sparse or Lazy; and set the quantum computer to use that method.
        '''
        if matrix_type == "Dense":
            self.Matrix = DenseMatrix
        if matrix_type == "Sparse":
            self.Matrix = SparseMatrix
        if matrix_type == "Lazy":
            self.Matrix = LazyMatrix
        if matrix_type == "LazySingle":
            self.Matrix = LazyMatrixSingle

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
        self.binary = self.__produce_digits()

        # gate inputs
        self.single_inputs = ["H", "P", "X", "Y", "Z", "M0", "M1"]
        self.matrices = [self.H, self.P, self.X, self.Y, self.Z, self.M0, self.M1]

        self.double_inputs = ["CV", "CNOT", "CZ"]

        #Initalise empty list to store history of gates used.
        self.__gate_history = []
        self.__custom_gate_names = []
        self.circuit = MatrixFrame

    def print_circuit(self):
        '''
        Prints the circuit that is built from using `add_gate_to_circuit` function.

        WARNING: Need to call `print_circuit_ascii` from terminal/cmd and will clear the terminal screen.
            Prints the quantum circuit in an ascii format on the terminal.
        '''
        pc = PrintingCircuit(self.__gate_history, self.N, custom_gate_names=self.__custom_gate_names)
        pc.print_circuit_ascii()

    def __produce_digits(self):  # this is the flipped basis, working on
        '''
        This produces the binary basis needed for the register.
        '''
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

    def __single_gates(self, gate: list, qnum: list) -> MatrixFrame:
        '''
        Build the single input gates to a 2^N matrix where N is the number of Qubits.

        <b>gate</b> A list of gates to build. <br>
        <b>qnum</b> List of the number of qubits corresponding to each gate.<br>
        <b>return</b> A gate as the specified matrix object.
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

    def __double_gates(self, gate: list, qnum: list) -> MatrixFrame:
        '''
        Use the Matrix method in the given desired method to build a multi-input gate.

        <b>gate</b> A list with one elemnt being the double gate. <br>
        <b>qnum</b> Qubit list. <br>
        <b>return</b> Returns correctly sized multi-input gate.
        '''
        if gate[0] == "CV":
            return self.Matrix("CV", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.Matrix("CNOT", self.binary, qnum[0][0], qnum[0][1])
        if gate[0] == "CZ":
            return self.Matrix("CZ", self.binary, qnum[0][0], qnum[0][1])

    def __validate_gate_logic_inputs(self, inputs: list):
        '''
        A custom check function to verify the list of time steps when the user is adding gates to the circuit.
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

    def gate_logic(self, inputs: list, add_gate_name: str = "") -> MatrixFrame:
        '''
        Builds the quantum circuit from a list of time steps. See README.md for examples of what to enter into the parameter `input`.

        <b>inputs</b> List of time steps. <br>
        <b>add_gate_name</b> Name of gates to be added. <br>
        <b>return</b> The circuit of type determined from the instantiation of quantum computer.
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
                M.append(self.__single_gates(inputs[i][0], inputs[i][1]))
            elif all(gate_double) == True:
                M.append(self.__double_gates(inputs[i][0], inputs[i][1]))
            else:
                print(
                    "Input error: single gates and double gates must be in separate steps. Returning identity matrix instead.")
                return self.Matrix('I')

        M = np.flip(M, axis=0)
        m = M[0]
        for i in range(1, len(M)):
            m = self.Matrix.matrix_multiply(m, M[i])

        return m

    def measure_any(self, qnum: int, state: int, register: list) -> int:
        '''
        Generate the measurement of the quantum circuit; once measured the system's wave function is collapsed.

        <b>qnum</b> Number of qubits. <br>
        <b>state</b> State of the qubit. <br>
        <b>state</b> Result of the measurement of qubit qnum.
        '''
        check.check_type(qnum, int)
        check.check_type(state, int)
        check.check_type(register, list)

        inner_register = self.Matrix.inner_product(register)
        if state == 0:
            matrix = self.__single_gates(["M0"], [[qnum]])
        elif state == 1:
            matrix = self.__single_gates(["M1"], [[qnum]])

        QP = self.Matrix.trace(self.Matrix.matrix_multiply(matrix, inner_register))
        #print('QP= ', QP)
        if (np.random.rand() < QP):
            result = 0
        else:
            result = 1
        return result

    def apply_register_and_measure(self, repeats: int = 1000, user_input_vector: list = []) -> dict:
        '''
        Apply's a register to the circuit built with `add_gate_to_circuit`, with default being the |0> state in the computatinal basis.

        <b>repeats</b> The number of measurements to be taken; assuming upon each measurement a new register is applied. <br>
        <b>user_input_vector</b> Allows custom choice of a register. <br>
        <b>returns</b> A dictionary of keys labelling the binary states, and the values the number of times that state was measured over.
        '''
        check.check_type(repeats, int)

        # Checks to see if user has defined their own register. If so, switch to theirs and validate
        input_vector = np.zeros(2**self.N)
        input_vector[0] = 1
        if not user_input_vector == []:
            check.check_type(user_input_vector, list)
            check.check_array_shape(user_input_vector, (2**self.N))
            check.check_sum(user_input_vector, 1.00) # Check normalisation.
            input_vector = user_input_vector

        probabilities = self.circuit.apply_register(input_vector) #Get probabilities from applying input vector

        binary_states = {}
        for i, basis in enumerate(self.binary):
            binary_states[''.join([str(j) for j in basis[::-1]])] = 0 # Creates the binary label, such as 001 for |001>.


        keys_list = list(binary_states.keys())
        for _ in range(repeats): # Repeats the measurements a number of times
            cumulative = 0
            skip = False
            j = 0
            random_var = random.random()
            while (not skip) and j < len(probabilities): # From random number between 0 and 1, finds the component of the vector by cummulative probability.
                cumulative += probabilities[j]
                if cumulative > random_var:
                    binary_states[keys_list[j]] += 1
                    skip = True
                j += 1

        return binary_states

    def add_gate_to_circuit(self, inputs: list, add_gate_name:str = ""):
        '''
        Adds the gates to the class ready to for building the circuit later. See README.md for examples of what to enter into the paramter `input`.

        <b>inputs</b> A list of time steps. <br>
        <b>add_gate_name</b> A label to replace the list of time steps with when the circuit is printed to the terminal using `print_circuit`.<br>
        '''
        check.check_type(add_gate_name, str)
        self.__validate_gate_logic_inputs(inputs)

        if not add_gate_name == "": # If defining a gate with a custom Name
            length_of_gate_history = len(self.__gate_history)
            self.__custom_gate_names.append(
                [length_of_gate_history-1, length_of_gate_history + len(inputs), add_gate_name]
            )
        self.__gate_history = self.__gate_history + inputs

    def build_circuit(self) -> MatrixFrame:
        '''
        Builds the circuit from the gates added via `add_gate_to_circuit` function.

        <b>return</b> The object specified by the instantiation of the class Dense/Sparse/Lazy object.
        '''
        self.circuit = self.gate_logic(self.__gate_history)
        return self.circuit
