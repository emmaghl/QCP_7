from QuantumComputer import Quantum_Computer

def main():
    qc = Quantum_Computer(3)

    # An example with an undefined gate G, repeating two hadamard gates (2nd element) and a bunch of others.
    gates = [(["H"], [[2]]), (["H", "H"], [[2], [1]]), (["CNOT"], [[1, 2]]),
         (["X"], [[1]]), (["CNOT"], [[1, 0]]), (["CV"], [[2, 1]]), (["CNOT"], [[0, 2]]),
         (["G"], [[0, 2]])]

    circuit = qc.Gate_Logic(gates)

    qc.print_circuit()

    print("With the matrix representation:\n")
    print(circuit)

if __name__=="__main__":
    main()