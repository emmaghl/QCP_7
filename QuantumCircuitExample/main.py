from QuantumComputer import Quantum_Computer
import numpy as np

def glue_circuits(matricies: list[np.ndarray]) -> np.ndarray:
    '''
    Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.
    '''
    m = np.identity(len(matricies[0]))
    for matrix in np.flip(matricies, axis=0):
        m = np.matmul(m, matrix)
    return m

def example():
    ''' An example of implementing a 3 qubit system, using dense matricies, and then printing the circuit to the terminal.'''
    qc = Quantum_Computer(3)

    # Defines toffoli gate
    toffoli_steps = [
        (["H"], [[2]]), (["CV"], [[1, 2]]),
        (["H"], [[1]]), (["CV"], [[0, 1]]), (["CV"], [[0, 1]]), (["H"], [[1]]),
        (["CV"], [[1, 2]]), (["CV"], [[1, 2]]), (["CV"], [[1, 2]]),
        (["H"], [[1]]), (["CV"], [[0, 1]]), (["CV"], [[0, 1]]), (["H"], [[1]]),
        (["CV"], [[0, 2]]), (["H"], [[2]])
    ]

    # An example with an undefined gate G, repeating two hadamard gates (2nd element) and a bunch of others.
    gates = [(["H"], [[2]]), (["H", "H"], [[2], [1]]), (["CNOT"], [[1, 2]]),
             (["X"], [[1]]), (["CNOT"], [[1, 0]]), (["CV"], [[2, 1]]), (["CNOT"], [[0, 2]]),
             (["G"], [[0, 2]])]

    circuits = [
        qc.Make_Gate_Logic(toffoli_steps, "To"),
        qc.Gate_Logic(gates),
        qc.Make_Gate_Logic(toffoli_steps, "To")
    ]
    qc.print_circuit()

    print("With the matrix representation:")
    print(glue_circuits(circuits))

def GroverAlgorithm():
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = Quantum_Computer(2)

    # Defines the gates for grover's algorithm
    init_states = [(["H", "H"], [[0], [1]])]

    oracle = [(["CZ"], [[0, 1]])]

    amplify_amplitude = [
            (["H", "H"], [[0], [1]]),
            (["X", "X"], [[0], [1]]),
            (["CZ"], [[1, 0]]),
            (["X", "X"], [[0], [1]]),
            (["H", "H"], [[0], [1]])
    ]

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = [
        qc.Gate_Logic(init_states),
        qc.Gate_Logic(oracle),
        qc.Gate_Logic(amplify_amplitude)
    ]

    # Prints circuit and matrix.
    qc.print_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = glue_circuits(circuits)
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput state vector:")
    print(glued_circuits.dot([1, 0, 0, 0]))

def main():
    #GroverAlgorithm()
    example()

if __name__=="__main__":
    main()

