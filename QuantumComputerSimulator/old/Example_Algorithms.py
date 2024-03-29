from QuantumComputer import QuantumComputer

def example():
    ''' An example of implementing a 3 qubit system, using dense matricies, and then printing the circuit to the terminal.'''

    qc = QuantumComputer(3)

    qc.add_single_gate(-np.pi/2, 0, 0, "Rz2N")

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
        #qc.Sparse_Gate_Logic(toffoli_steps, "Tof"),
        qc.Gate_Logic(gates)
        #qc.Make_Gate_Logic(toffoli_steps, "To")
    ]
    #qc.print_circuit()

    print("With the matrix representation:")
    print(glue_circuits(circuits))

def CCnot(control_1, control_2, target) -> np.array:
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

def CCCnot(control_1, control_2, control_3, target, auxilary) -> np.array:
    return glue_lists(
        CCnot(control_1, control_2, auxilary),
        CCnot(control_3,auxilary, target),
        CCnot(control_1, control_2, auxilary)
    )

def CCz(control_1, control_2, target) -> np.array:
    return [
        (["CNOT"], [[control_2, target]]),
        (["TD"], [[target]]),
        (["CNOT"], [[control_1, target]]),
        (["T"], [[target]]),
        (["CNOT"], [[control_2, target]]),
        (["TD"], [[target]]),
        (["CNOT"], [[control_1, target]]),
        (["T"], [[control_2]]),
        (["T"], [[target]]),
        (["CNOT"], [[control_1, control_2]]),
        (["T"], [[control_1]]),
        (["TD"], [[control_2]]),
        (["CNOT"], [[control_1, control_2]])
    ]

def make_C_gate(alpha: float, theta: float, beta: float, qc: QuantumComputer, control, target, name: str):
    qc.add_single_gate(alpha, 0, 0, name+"1")
    qc.add_single_gate(0, 0, theta/2, name+"2")
    qc.add_single_gate(0, 0, -theta/2, name+"3")
    qc.add_single_gate(-(alpha+beta)/2, 0, 0, name+"4")
    qc.add_single_gate((beta-alpha)/2, 0, 0, name+"5")
    new_gate = [
        ([name+"1"], [[target]]),
        ([name+"2"], [[target]]),
        (["CNOT"], [[control, target]]),
        ([name+"3"], [[target]]),
        ([name+"4"], [[target]]),
        (["CNOT"], [[control, target]]),
        ([name+"5"], [[target]])
    ]
    return new_gate

def GroverAlgorithm_Suduko():
    '''The paper arxiv:9503016 by Barenco. A et al. is used to build multi-control gates.'''
    num_qubits = 11
    qc = QuantumComputer(num_qubits)


    # Defines the gates for grover's algorithm
    init_states = [
        qc.Gate_Logic([ (["H", "H", "H", "H", "H", "H", "X"], [[0], [1], [2], [3], [4], [5], [9]]) ]),
        qc.Gate_Logic([ (["H"], [[9]]) ])
    ]

    classical_chossing = [
        qc.Make_Gate_Logic(CCnot(1, 2, 4), "T4"),
        qc.Make_Gate_Logic(CCnot(0, 3, 8), "T"),
        qc.Make_Gate_Logic(CCnot(1, 4, 8), "T"),
        qc.Make_Gate_Logic(CCnot(2, 5, 8), "T"),
        qc.Make_Gate_Logic(CCCnot(0, 1, 2, 6, 10), "T4"),
        qc.Gate_Logic([
            (["CNOT"], [[0, 6]]),
            (["CNOT"], [[1, 6]]),
            (["CNOT"], [[2, 6]]),
        ]),
        qc.Make_Gate_Logic(CCCnot(3, 4, 5, 7, 10), "T4"),
        qc.Gate_Logic([
            (["CNOT"], [[3, 7]]),
            (["CNOT"], [[4, 7]]),
            (["CNOT"], [[5, 7]]),
        ])
    ]

    kick_back = [
        qc.Make_Gate_Logic(CCnot(1, 2, 4), "T4"),
        qc.Make_Gate_Logic(CCCnot(6, 7, 8, 9, 10), "T4")
    ]

    reset = [qc.Make_Gate_Logic(CCnot(1, 2, 4), "T4"),
             qc.Make_Gate_Logic(CCnot(0, 3, 8), "T"),
             qc.Make_Gate_Logic(CCnot(1, 4, 8), "T"),
             qc.Make_Gate_Logic(CCnot(2, 5, 8), "T"),
             qc.Make_Gate_Logic(CCCnot(0, 1, 2, 6, 10), "T4"),
             qc.Gate_Logic([
                 (["CNOT"], [[0, 6]]),
                 (["CNOT"], [[1, 6]]),
                 (["CNOT"], [[2, 6]])
             ]),
             qc.Make_Gate_Logic(CCCnot(3, 4, 5, 7, 10), "T4"),
             qc.Gate_Logic([
                 (["CNOT"], [[3, 7]]),
                 (["CNOT"], [[4, 7]]),
                 (["CNOT"], [[5, 7]])
             ])

             ]



    amplify_amplitude = [
        qc.Make_Gate_Logic(CCnot(1, 2, 4), "T4"),
        qc.Gate_Logic([ (["H", "H", "H", "H", "H", "H"], [[0], [1], [2], [3], [4], [5]]) ]),
        qc.Gate_Logic([ (["X", "X", "X", "X", "X", "X"], [[0], [1], [2], [3], [4], [5]]) ]),
        qc.Gate_Logic([ (["CZ"], [[0, 5]]) ]),
        qc.Gate_Logic([ (["H", "H", "H", "H", "H", "H"], [[0], [1], [2], [3], [4], [5]]) ]),
        qc.Gate_Logic([ (["X", "X", "X", "X", "X", "X"], [[0], [1], [2], [3], [4], [5]]) ])

    ]

    oracle = glue_lists(classical_chossing, kick_back, reset, classical_chossing)

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = glue_lists(init_states, oracle, amplify_amplitude)
    # Prints circuit and matrix.
    qc.print_circuit()



    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = qc.Sparse_to_Dense(circuits[0])
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    startVec = np.zeros((2**num_qubits))
    startVec[0]=1
    outVec = glued_circuits.dot(startVec)
    print(np.array([outVec[i]*np.conjugate(outVec[i]) for i in range(len(outVec))]))

def GroverAlgorithm_Mini_Suduko():
    num_qubits = 6
    qc = QuantumComputer(num_qubits)

    # Defines the gates for grover's algorithm
    init_states = [
        qc.Gate_Logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
        qc.Gate_Logic([ (["H"], [[4]]) ]),
        qc.Gate_Logic([ (["X"], [[4]]) ])
    ]

    classical_chossing = [
        qc.Make_Gate_Logic(CCCnot(0, 1, 2, 3, 5), "T4"),
        qc.Gate_Logic([
            (["CNOT"], [[0, 3]]),
            (["CNOT"], [[1, 3]]),
            (["CNOT"], [[2, 3]]),
        ])
    ]

    kick_back = [
        qc.Gate_Logic([
            (["CNOT"], [[3, 4]]),
        ])
    ]

    reset = [
        qc.Make_Gate_Logic(CCCnot(0, 1, 2, 3, 5), "T4"),
        qc.Gate_Logic([
            (["CNOT"], [[0, 3]]),
            (["CNOT"], [[1, 3]]),
            (["CNOT"], [[2, 3]]),
        ])
    ]


    amplify_amplitude = [
        qc.Gate_Logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
        qc.Gate_Logic([
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]])
        ]),
        qc.Gate_Logic(CCz(0, 1, 2), "Z"),
        qc.Gate_Logic([
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]])
        ]),
        qc.Gate_Logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
    ]

    oracle = glue_lists(classical_chossing, kick_back, reset)

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = glue_lists(init_states, oracle, amplify_amplitude)
    # Prints circuit and matrix.
    qc.print_circuit()

    glued_circuits = glue_circuits(circuits)
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    startVec = np.zeros((2**num_qubits))
    startVec[0]=1
    outVec = glued_circuits.dot(startVec)
    print(np.array([100*outVec[i]*np.conjugate(outVec[i]) for i in range(len(outVec))], dtype=np.float32))

def testing():
    qc = QuantumComputer(1)

    # Defines the gates for grover's algorithm
    circuits = [
        qc.Gate_Logic([
            (["Y"], [[0]]), (["Z"], [[0]])
        ])
    ]

    # Prints circuit and matrix.
    #qc.print_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = glue_circuits(circuits, qc)

    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    outVec = np.matmul(glued_circuits,[1, 0])
    #print(np.array([100*outVec[i]*np.conjugate(outVec[i]) for i in range(len(outVec))], dtype=np.float32))
    print(np.array([outVec[i] for i in range(len(outVec))]))


def GroverAlgorithm_2Qubit():
    qc = Quantum_Computer(2)

    # Defines the gates for grover's algorithm
    init_states = [(["H", "H"], [[0], [1]])
                   ]

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
    print(glued_circuits.shape)
    print(glued_circuits.dot([1, 0, 0, 0]))
