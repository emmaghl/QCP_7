'''
Sample file for testing the QuantumComputerSimulator package, and contains the Grover algorithms as described in the README.md. This file can also be used as an example for how the user interfaces with the QuantumComputerSimulator package.

To run tests, add `--test` argument when running from the terminal.
'''
from QuantumComputerSimulator import QuantumComputer, Test

import matplotlib.pyplot as plt
import sys
import numpy as np

def user_validation(msg: str, options: list[str]) -> str:
    '''Validates the user input from the terminal.'''
    print(msg)
    user_input = input('>')
    while not user_input.lower() in options:
        print(f'Please select from: {options}.')
        print(msg)
        user_input = input('>')
    return user_input.lower()

def CCnot(control_1, control_2, target) -> list:
    '''Defines the Toffoli gate, and an example for implementing other gates from the elementary ones.'''
    gate_built = [
        (["H"], [[target]]),
        (["CV"], [[control_2, target]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_2, target]]),
        (["CV"], [[control_2, target]]),
        (["CV"], [[control_2, target]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, target]]),
        (["H"], [[target]])
    ]
    return gate_built

def CCCnot(control_1, control_2, control_3, target, auxilary) -> list:
    '''CCCNot gate.'''
    return CCnot(control_1, control_3, auxilary) + CCnot(control_2,auxilary, target) +CCnot(control_1, control_3, auxilary) #Python for concatenating lists

def GroverAlgorithm_3Qubit(matrixtype, show_plots=False):
    '''A function implementing a three qubit version of Grover's algorithm. Only the states |101> and |111> should be measured.'''
    qc = QuantumComputer(3, matrixtype)

    # Defines the gates for grover's algorithm
    init_states = [
        (["H", "H", "H"], [[0], [1], [2]])
    ]

    oracle = [
        (["CZ"], [[0, 2]])
    ]

    half_of_amplification = [
        (["H", "H", "H"], [[0], [1], [2]]),
        (["X", "X", "X"], [[0], [1], [2]]),
        (["H"], [[2]])
    ]

    # Feeds the gates that the circuit will be built out of. This is order dependent
    qc.add_gate_to_circuit(init_states),
    qc.add_gate_to_circuit(oracle),
    qc.add_gate_to_circuit(half_of_amplification),
    qc.add_gate_to_circuit(CCnot(0, 1, 2), add_gate_name="T"),
    qc.add_gate_to_circuit(half_of_amplification[::-1]) # Reverses list

    # Prints circuit and make's sure the user is happy with it before it's built.
    qc.print_circuit()
    user = user_validation('Continue building with circuit? (y/n)', ['y', 'n'])
    if user == 'n':
        exit()

    # Builds the circuit using matrix methods
    circuit = qc.build_circuit()

    # Prints the matrix representation of the circuits, as it is using Dense techniques, the circuit will be represented as a matrix.
    print("With the matrix representation:")
    print(circuit.matrix)

    # The register is set to be |000>, and the states that amplified should be |101> and |111>
    print("\nBin count of binary states after 1000 runs:")
    probs = qc.apply_register_and_measure(repeats=1000)
    [print(f"|{i}> : {probs[i]}") for i in probs.keys()]

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

def GroverAlgorithm_SingleRow_BinaryCol_Sudoku(matrixtype, show_plots = False):
    '''The smaller version of the 3x3 single roq sudoko, in the sense that this checks one binary column.'''
    # The roles of each qubit are:
    #   - 0, 1, 2: the qubits that are amplified
    #   - 3: this qubit handles the signal that 'turns on' the phase kickback (in a classical sense)'
    #   - 4: In the |-> state, that implements the phase kickback.
    #   - 5: Garbage qubit for the CCCnot gate to work.
    num_qubits = 6
    qc = QuantumComputer(num_qubits, matrixtype)

    # Initialises by putting three qubits in a super position of equal weight, and the fourth qubit in the |-> state to implement phase kick-back.
    init_states = [
        (["H", "H", "H"], [[0], [1], [2]]),
        (["X"], [[4]]),
        (["H"], [[4]])
    ]
    qc.add_gate_to_circuit(init_states)

    # Builds the oracle
    qc.add_gate_to_circuit(CCCnot(0, 1, 2, 3, 5), add_gate_name="T4") # A 3 controlled NOT gate (an extended version of the Toffoli gate)
    oracle_continued = [
        (["CNOT"], [[0, 3]]),
        (["CNOT"], [[1, 3]]),
        (["CNOT"], [[2, 3]])
    ]
    qc.add_gate_to_circuit(oracle_continued)

    # This gate is the only one that links to the 5th qubit, implementing the phase kickback.
    qc.add_gate_to_circuit([
        (["CNOT"], [[3, 4]]),
    ])

    # Reset the 4th qubit, by repeating the oracle. Funcrtions have to be called explicilty to aff the oravle
    qc.add_gate_to_circuit(CCCnot(0, 1, 2, 3, 5), add_gate_name="T4") # A 3 controlled NOT gate (an extended version of the Toffoli gate)
    reset_continued = [
        (["CNOT"], [[0, 3]]),
        (["CNOT"], [[1, 3]]),
        (["CNOT"], [[2, 3]])
    ]
    qc.add_gate_to_circuit(reset_continued)

    # Amplify the amplitudes
    amplify_amplitude = [
        (["H", "H", "H"], [[0], [1], [2]]),
        (["X", "X", "X"], [[0], [1], [2]]),
        (["H"], [[2]])
    ]
    qc.add_gate_to_circuit(amplify_amplitude)
    qc.add_gate_to_circuit(CCnot(0, 1, 2), "Z")
    qc.add_gate_to_circuit(amplify_amplitude[::-1])

    # Prints circuit and make's sure the user is happy with it before it's built. Especially useful here, as this will take a bit of time.
    qc.print_circuit()
    user = user_validation('Continue building with circuit? (y/n)', ['y', 'n'])
    if user == 'n':
        exit()

    # Once the user has verified that the circuit digram is the one intedned, then starts to build it
    qc.build_circuit()

    # Only selects the non-zero bin counts
    print("\nBin count of binary states after 1000 runs:")
    probs = qc.apply_register_and_measure(repeats=1000)
    for i in probs.keys():
        state_probs = probs[i]
        if not state_probs == 0:
            print(f"|{i}> : {probs[i]}")

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

def GroverAlgorithm_SingleRow_Sudoku(matrixtype, show_plots = False):
    '''The smaller version of the 3x3 single roq sudoko, in the sense that this checks one binary column.'''
    # The roles of each qubit are:
    #   - 0 to 5: the qubits that are amplified
    #   - 6, 7, 8: this qubit handles the signal that 'turns on' the phase kickback (in a classical sense)'
    #   - 9: In the |-> state, that implements the phase kickback.
    #   - 10: Garbage qubit for the CCCnot gate to work.
    num_qubits = 11
    qc = QuantumComputer(num_qubits, matrixtype)

    # Initialises by putting three qubits in a super position of equal weight, and the fourth qubit in the |-> state to implement phase kick-back.
    init_states = [
        (["H", "H", "H", "H", "H", "H"], [[0], [1], [2], [3], [4], [5]]),
        (["X", "X"], [[9], [8]]),
        (["H"], [[9]])
    ]
    qc.add_gate_to_circuit(init_states)

    # Builds the oracle
    qc.add_gate_to_circuit(CCnot(0, 3, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCnot(1, 4, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCnot(2, 5, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCCnot(0, 1, 2, 6, 10), add_gate_name="T4") # A 3 controlled NOT gate (an extended version of the Toffoli gate)
    qc.add_gate_to_circuit([
        (["CNOT"], [[0, 6]]),
        (["CNOT"], [[1, 6]]),
        (["CNOT"], [[2, 6]])
    ])
    qc.add_gate_to_circuit(CCCnot(3, 4, 5, 7, 10), add_gate_name="T4")
    qc.add_gate_to_circuit([
        (["CNOT"], [[3, 7]]),
        (["CNOT"], [[4, 7]]),
        (["CNOT"], [[5, 7]])
    ])

    # The phase kickback
    qc.add_gate_to_circuit(CCCnot(6, 7, 8, 9, 10), add_gate_name="T4")

    # Reset, byusing the oracle again
    qc.add_gate_to_circuit(CCnot(0, 3, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCnot(1, 4, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCnot(2, 5, 8), add_gate_name="T")
    qc.add_gate_to_circuit(CCCnot(0, 1, 2, 6, 10), add_gate_name="T4") # A 3 controlled NOT gate (an extended version of the Toffoli gate)
    qc.add_gate_to_circuit([
        (["CNOT"], [[0, 6]]),
        (["CNOT"], [[1, 6]]),
        (["CNOT"], [[2, 6]])
    ])
    qc.add_gate_to_circuit(CCCnot(3, 4, 5, 7, 10), add_gate_name="T4")
    qc.add_gate_to_circuit([
        (["CNOT"], [[3, 7]]),
        (["CNOT"], [[4, 7]]),
        (["CNOT"], [[5, 7]])
    ])

    # Amplify the amplitudes
    amplify_amplitude = [
        (["H", "H", "H", "H", "H", "H"], [[0], [1], [2], [3], [4], [5]]),
        (["X", "X", "X", "X", "X", "X"], [[0], [1], [2], [3], [4], [5]]),
        (["H"], [[3]])
    ]
    qc.add_gate_to_circuit(amplify_amplitude)
    qc.add_gate_to_circuit(CCCnot(0, 1, 2, 3, 10), "Z")
    qc.add_gate_to_circuit(amplify_amplitude[::-1])

    # Prints circuit and make's sure the user is happy with it before it's built. Especially useful here, as this will take a bit of time.
    qc.print_circuit()
    user = user_validation('Continue building with circuit? (y/n)', ['y', 'n'])
    if user == 'n':
        exit()

    qc.build_circuit() # Builds circuit

    print("\nBin count of binary states after 1000 runs:")
    probs = qc.apply_register_and_measure(repeats=1000)
    for i in probs.keys():
        state_probs = probs[i]
        if not state_probs == 0:
            print(f"|{i}> : {probs[i]}")

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

if __name__=="__main__":
    # Runs example algorithms if not testing contents
    if len(sys.argv) == 1:
        options = [
            '[1] 3 qubit Grovers',
            '[2] Single binary row of 3x3 sudoko',
            '[3] A full row of 3x3 sudoko',
        ]
        [print(f'{o}') for o in options]

        user_input = user_validation('Enter the number beside the algorithm that you would like to run.', ['1', '2', '3', 'exit'])

        if user_input == 'exit':
            exit()

        matrix_input = user_validation(
            'Enter type of matrix to be used with your chosen algorithm. Type D for Dense, S for Sparse, L for Lazy, LS for Single Lazy',
            ['d', 's', 'l', 'ls'])
        matrix_type = ''
        if matrix_input == 'd':
            matrix_type = 'Dense'
        elif matrix_input == 's':
            matrix_type = 'Sparse'
        elif matrix_input == 'l':
            matrix_type = 'Lazy'
        elif matrix_input == 'ls':
            matrix_type = 'LazySingle'

        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)# Makes output from numpy arrays more pleasant.
        print('Running, this may take a bit...')

        if user_input == '1':
            GroverAlgorithm_3Qubit(matrix_type, show_plots=False)
        elif user_input == '2':
            GroverAlgorithm_SingleRow_BinaryCol_Sudoku(matrix_type, show_plots=False)
        elif user_input == '3':
            GroverAlgorithm_SingleRow_Sudoku(matrix_type, show_plots=False)

    elif '--test' in sys.argv: # Running the tests via Test class.
        print("Running tests...")
        Test()
        print("Tests completed successfully.")
        exit()
    else:
        print("The only argument available is '--test'. \nThis runs all functions in the class 'Test' (that don't have _ in front of their name). \nThe 'Tests' class is located in QuantumComputerSimulator/Tests.py.")