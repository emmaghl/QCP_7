import numpy as np
class Quantum_Computer:

	def __init__(self, Qubits):
		self.Register_Size = Qubits
		self.X = Gates.X()
		self.Y = Gates.Y()
		self.Z = Gates.Z()
		self.H = Gates.Hadamard()
		self.RN = Gates.Root_Not()

	
	def Register():
		def Tensor_Prod():
			pass
	
	
class Gates:
	def __init__(self):
		pass
	def X():
		return np.array([[0,1],[1,0]])
	def Y():
		return Y = np.array([[0,0+1j],[0-1j,0]],dtype = complex)
	def Hadamard():
		return np.array([[1,1],[1,-1]])
