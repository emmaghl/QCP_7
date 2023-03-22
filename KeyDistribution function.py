from QuantumComputerSimulator import QuantumComputer, Test
from QuantumComputerSimulator.mods.SparseMatrix import SparseMatrix
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix
from QuantumComputerSimulator.mods.LazyMatrix import LazyMatrix

import numpy as np
import random
import time
import matplotlib.pyplot as plt



def key_distr(matrix_type, qnum, listening):
    t = matrix_type # "D", "S", "L"
    n = qnum # int
    y = listening #

    global qc
    if t == "D" or t == "d":
        qc = QuantumComputer(n, 'Dense')
    if t == "S" or t == "s":
        qc = QuantumComputer(n, 'Sparse')
    if t == "L" or t == "l":
        qc = QuantumComputer(n, 'Lazy')

    np.random.seed(13)

    # Step 0 ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    start_1 = time.time()
    register = qc.Matrix.quantum_register(n)

    #print('Step 0 complete: Qubit register setup')

    # Step 1 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    np.random.seed(13)
    A_bits = np.random.randint(2, size=n)

    #print('Step 1 complete:', 'A bits =', A_bits, '!This is not shared publicly!')

    #Step 2 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    np.random.seed(13)
    A_bases = np.random.randint(2, size=n)

    #print('Step 2 complete:', 'A bases =', A_bases, '!This is not shared publicly!')
    #Step 3 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    for i in range(n):
        j = A_bases[i]
        k = A_bits[i]
        if j == 0:
            if k == 0:
                pass
            else:
                circuit = qc.gate_logic([(["X"], [[i]])])
                circuit = circuit.matrix
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix
        if j == 1:
            if k == 0:
                circuit = qc.gate_logic([(["H"], [[i]])])
                circuit = circuit.matrix
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix
            else:
                circuit_1 = qc.gate_logic([(["X"], [[i]])])
                circuit_1 = circuit_1.matrix
                circuit_2 = qc.gate_logic([(["H"], [[i]])])
                circuit_2 = circuit_2.matrix
                circuit = qc.Matrix.matrix_multiply(circuit_2, circuit_1)
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix


    # print('Step 3 complete: Qubits encoded')

    end_1 = time.time()
    time_1 = end_1 - start_1
    #print('Time_1 =', time_1)

    # Step Interception ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    while True:
        if y == "Yes" or y == "yes" or y == "YES":
            start_2 = time.time()
            np.random.seed(13)
            C_bases = np.random.randint(2, size=n)

            register_intercept = []
            for i in range(n):
                g = C_bases[i]
                if g == 0:
                    result = qc.measure_any(i, 0, register)
                    if qc.Matrix == DenseMatrix:
                        register_intercept.append(result)
                    elif qc.Matrix == SparseMatrix:
                        register_intercept.append([0,i,result])
                    elif qc.Matrix == LazyMatrix:
                        break
                else:
                    circuit = qc.gate_logic([(["H"], [[i]])])
                    circuit = circuit.matrix

                    register = qc.Matrix.matrix_multiply(circuit, register)
                    register = register.matrix
                    result = qc.measure_any(i, 0, register)
                    if qc.Matrix == DenseMatrix:
                        register_intercept.append(result)
                    elif qc.Matrix == SparseMatrix:
                        register_intercept.append([0,i,result])
                    elif qc.Matrix == LazyMatrix:
                        break

            zero_col = qc.Matrix("zerocol")
            one_col = qc.Matrix("onecol")

            for i in range(n):
                q = register_intercept[i]
                if i == 0:
                    if q == 0:
                        register = qc.Matrix.transpose(zero_col)
                    else:
                        register = qc.Matrix.transpose(one_col)
                else:
                    if q == 0:
                        register = qc.Matrix.tensor_prod(zero_col, register)
                        register = register.matrix
                    else:
                        register = qc.Matrix.tensor_prod(one_col, register)
                        register = register.matrix
            break
        elif y == "No" or y == "no" or y == "NO":
            start_2 = time.time()
            break
    # Step 4 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    np.random.seed(13)
    B_bases =  np.random.randint(2, size=n)

    measurement = []
    for i in range(n):
        g = B_bases[i]
        if g == 0:

            result = qc.measure_any(i, 0, register)
            measurement.append(result)

        else:
            circuit = qc.gate_logic( [(["H"], [[i]])] )
            circuit = circuit.matrix
            register = qc.Matrix.matrix_multiply(circuit, register)
            register = register.matrix
            result = qc.measure_any(i, 0, register)
            measurement.append(result)

    #print('Step 4 complete:')

    # Step 5 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    B_Key = []

    for i in range(n):
        a = A_bases[i]
        b = B_bases[i]
        c = measurement[i]
        if a == b:
            B_Key.append(c)
        else:
            pass

    A_Key = []
    for i in range(n):
        a = A_bases[i]
        b = B_bases[i]
        c = A_bits[i]
        if a == b:
            A_Key.append(c)
        else:
            pass


    #print('Step 5 complete')
    #print('A Key =', A_Key , 'This is not shared publicly')
    #print('B Key =', B_Key, 'This is not shared publicly')

    # Step 6 ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    sample_A = []
    sample_B = []

    j = len(A_Key)
    l = len(B_Key)


    number_of_samples = round(j*0.5)

    number_of_samples = 3

    s = random.sample(range(0, j), number_of_samples)

    #print (s)

    for i in s:
        f = A_Key[i]
        h = B_Key[i]
        sample_A.append(f)
        sample_B.append(h)


    # Step 7 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    j = len(sample_A)
    l = len(sample_B)

    for i in range(j):
        p = 1
        if sample_A[i] == sample_B[i]:
            p = p + 1
        else:
            end_2 = time.time()
            time_2 = end_2 - start_2
            total_time = time_1 + time_2


    #print('Step 7 complete:', p)

    # Step 8 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    j = len(A_Key)
    t = range(0, j)
    t = list(t)

    for i in s:
        t.remove(i)

    A_secret_key = []
    B_secret_key = []

    for i in t:
        A_secret_key.append(A_Key[i])
        B_secret_key.append(B_Key[i])

    end_2 = time.time()
    time_2 = end_2 - start_2
    total_time = time_1 + time_2

    return total_time


qnum_list_D = range(3, 8)
qnum_list_S = range(3, 13)

t = "D"

y = "yes"
time_D_yes = []
for n in qnum_list_D:
    time_D_yes.append(key_distr(t, n, y))

print("done Dense, listening")

y = "no"
time_D_no = []
for n in qnum_list_D:
    time_D_no.append(key_distr(t, n, y))

print("done Dense, not listening")

t = "S"

y = "yes"
time_S_yes = []
for n in qnum_list_S:
    time_S_yes.append(key_distr(t, n, y))

print("done Sparse, listening")

y = "no"
time_S_no = []
for n in qnum_list_S:
    time_S_no.append(key_distr(t, n, y))

print("done Sparse, listening")

plt.plot(qnum_list_D, time_D_yes, color = "black", linestyle='solid', label = "dense, listening")
plt.plot(qnum_list_D, time_D_no, color = "black", linestyle='dashed', label = "dense, not listening")
plt.plot(qnum_list_S, time_S_yes, color = "darkgrey", linestyle='solid', label = "sparse, listening")
plt.plot(qnum_list_S, time_S_no, color = "darkgrey", linestyle='dashed', label = "sparse, not listening")

plt.xlabel("number of qubits")
plt.ylabel("time taken (s)")
plt.legend()
plt.show()