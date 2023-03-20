'''Sample file for testing the QuantumComputerSimulator module, and showcasing the features. To run tests, add `--test` argument when running from the terminal.'''
from QuantumComputerSimulator import QuantumComputer, Test

import matplotlib.pyplot as plt
import sys
import numpy as np

def glue_circuits(matricies: object) -> np.ndarray:
    ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
    m = matricies[0]

    for matrix in matricies[::-1]:
        m = matrix.matrix_multiply(m,matrix)
    return m

def user_validation(msg: str, options: list[str]) -> str:
    print(msg)
    user_input = input('>')
    while not user_input.lower() in options:
        print(f'Please select from: {options}.')
        print(msg)
        user_input = input('>')
    return user_input

def glue_lists(*lists) -> list:
    '''Adds elments from list_2 to list_1.'''
    big_list = []
    [[big_list.append(i) for i in list] for list in lists]
    return big_list

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
    return glue_lists(
        CCnot(control_1, control_3, auxilary),
        CCnot(control_2,auxilary, target),
        CCnot(control_1, control_3, auxilary)
    )

def GroverAlgorithm_3Qubit(show_plots=False):
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = QuantumComputer(3, 'Lazy')

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

    # Builds the circuit using 'Dense' methods
    circuit = qc.build_circuit()

    # Prints the matrix representation of the circuits, as it is using Dense techniques, the circuit will be represented as a matrix.
    print("With the matrix representation:")
    print(circuit.matrix)

    # The regiseter is set to be |000>, and the states that amplified should be |101> and |111>
    print("\nBin count of binary states after 1000 runs:")
    probs = qc.apply_register_and_measure(repeats=1000)
    [print(f"|{i}> : {probs[i]}") for i in probs.keys()]

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

def GroverAlgorithm_SingleRow_BinaryCol_Suduko(show_plots = False):
    '''The smaller version of the 3x3 single roq sudoko, in the sense that this checks one binary column.'''
    # The roles of each qubit are:
    #   - 0, 1, 2: the qubits that are amplified
    #   - 3: this qubit handles the signal that 'turns on' the phase kickback (in a classical sense)'
    #   - 4: In the |-> state, that implements the phase kickback.
    #   - 5: Garbage qubit for the CCCnot gate to work.
    num_qubits = 6
    qc = QuantumComputer(num_qubits, "Dense")

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

    qc.build_circuit() # Builds matrix

    print("\nBin count of binary states after 1000 runs:")
    probs = qc.apply_register_and_measure(repeats=1000)
    for i in probs.keys():
        state_probs = probs[i]
        if not state_probs == 0:
            print(f"|{i}> : {probs[i]}")

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

def GroverAlgorithm_SingleRow_Suduko(show_plots = False):
    '''The smaller version of the 3x3 single roq sudoko, in the sense that this checks one binary column.'''
    # The roles of each qubit are:
    #   - 0 to 5: the qubits that are amplified
    #   - 6, 7, 8: this qubit handles the signal that 'turns on' the phase kickback (in a classical sense)'
    #   - 9: In the |-> state, that implements the phase kickback.
    #   - 10: Garbage qubit for the CCCnot gate to work.
    num_qubits = 11
    qc = QuantumComputer(num_qubits, "Dense")

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
    # Runs example algorithms if not testing contents if not testing
    if len(sys.argv) == 1:
        # Prints the options to the user
        options = [
            '[1] 3 qubit Grovers (Dense)',
            '[2] Single binary row of 3x3 sudoko (Dense)',
            '[3] A full row of 3x3 sudoko (Dense)',
        ]
        [print(f'{o}') for o in options]
        user_input = user_validation('Enter the number beside the algorithm that you would like to run.', ['1', '2', '3', 'exit'])

        if user_input == 'exit':
            exit()

        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)# Makes output from numpy arrays more pleasant.
        print('Running, this may take a bit...')

        if user_input == '1':
            GroverAlgorithm_3Qubit(show_plots=False)
        elif user_input == '2':
            GroverAlgorithm_SingleRow_BinaryCol_Suduko(show_plots=False)
        elif user_input == '3':
            GroverAlgorithm_SingleRow_Suduko(show_plots=False)

    elif '--test' in sys.argv:
        print("Running tests...")
        Test()
        print("Tests completed successfully.")
        exit()
    else:
        print("The only argument available is '--test'. \nThis runs all functions in the class 'Test' (that don't have _ in front of their name). \nThe 'Tests' class is located in QuantumComputerSimulator/Tests.py.")