import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.X = Gates.X()
        self.Y = Gates.Y()
        self.Z = Gates.Z()
        self.H = Gates.Hadamard()
        self.CNot = Gates.C_not()

    def Register(self):
        def Tensor_Prod(self):
            pass


class Gates:
    def __init__(self, input):
        self.inpt = input

    def X(self):
        return np.matmul(np.array([[0, 1], [1, 0]]), self.inpt)

    def Y(self):
        return np.matmul(np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex),self.inpt)

    def Z(self):
        return np.matmul(np.array([[1, 0], [0, -1]]), self.inpt)

    def Root_Not(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]), self.inpt)

    def Hadamard(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), self.inpt)

    def Phase(self):
        return np.matmul(np.array([[1, 0], [1, 0 + 1j]], dtype=complex), self.inpt)

    def T(self):
        return np.matmul(np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex), self.inpt)

    def C_not(self):
        return np.matmul(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), self.inpt)

### Really not sure how to put this in ###

##Defining vector states##

Zero = np.array([[1.0],[0.0]])  #This is for the |0> vector state

One = np.array([[0.0],[1.0]])   #This is for the |1> vector state

##Defining qubit as superposition of eigenstate, Zero and One vector states##

a = 0.9 #These can be changed, just probabilities of finding in state, 

b = (1 - a**2)**0.5 #Normalizing condition as a**2 + b**2 = 1, but might want to make a "Normalize" function because I think this is sloppy

Qubit = a*Zero + b*One #I'm guessing this is a good way to make a superposition? But It should be complex and I haven't figured out how to do that yet 

print ("Qubit = ", Qubit) #just checking 

##Testing a Hadamard gate on different vector states, not sure how to apply it to the qubit? ##

Hadamard = 1./(2**0.5) * np.array([[1, 1], [1, -1]])

H_0 = np.dot(Hadamard, Zero)

H_1 = np.dot(Hadamard, One)

print ("H_0 =", H_0) #just checking 

print ("H_1 =", H_1) #just checking 



