#class QuantumKeyDistribution(Interface): ???

from QuantumComputerSimulator import QuantumComputer
import numpy as np
# np.random.seed(seed=0) ensures our random intiger starts from 0 


# Step 1 - generate a random bit string n long, this is the message, A_bits

# Step 2 - generate a random bit string n long, this is the corresponding, A_bases

# Step 3 - Set up each qubit corresponding to the combination of A_bases and A_bits

# Step 4 - B_measure, set up B_bases to measure_measure each qubit against that bases

# Step 5 - Compare the random choice of bases to achieve secret key of bits for both A and B! 

# Step 6 - Compare random sample of key to check, then discard, to see if there was any interception. 

def message_number(self):  #idk what to do here :/ #n = 10 for example, can we make this a variable? 
        n = self #?? 
        return n
        
n = ('Input number of bits: ')  #CREATE PARENT FUNCTION!!
        
# Step 1 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def A_bits(n):
        np.random.seed(seed=0) 
        A_bits = np.random.randint(2, size=n)  
        return np.random.randint(2, size=n)
#Step 2 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def A_bases(n):
        np.random.seed(seed=0) 
        A_bases = np.random.randint(2, size=n)  
        return np.random.randint(2, size=n)

#Step 3 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def A_encode(A_bases, A_bits):
        qc = QuantumComputer(n, "Dense") #Want to initialise the QC as all qubits in 0 state? How do I do this? 
        for i in range(n): 
                if A_bases[i] == 0 #Prepare in either |0> or |1> state dependent on bit message, this is Z - basis 
                        if A_bits[i] == 0 
                                pass #Only works id QC is initlaised to all qubits being 0, |0> 
                        else:    
                                qc.gate_logic( (["X"], [[i]]) )   # initialises to |1> 
                else: 
                        if A_bits[i] == 1  #Prepare in either |+> or |-> state dependent on bit message, this is Z - basis
                                qc.gate_logic( (["H"], [[i]]) )   # initialises to |-> 
                        else: 
                                qc.gate_logic( (["X"], [[i]]) )  
                                qc.gate_logic( (["H"], [[i]]) )   # initialises to |+> 
        code = qc
        return code
        
# Step 4 ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def B_bases(n): 
        np.random.seed(seed=0) 
        B_bases =  np.random.randint(2, size=n) 
        return np.random.randint(2, size=n)     
        
def B_measure(B_bases, code)
        measurement[]
        for i in range(n): 
                if B_bases[i] == 0 #measure against Z basis |0> or |1> 
                        measure_any(code, i, 0) # OR NOT!! #need to make changes to measure function! For different bases!! Like this atm is only for Z basis, measureing 0 or 1 will give identical results, need for + - as well                  
                else: 
                        code.gate_logic( (["H"], [[i]]) )  #Use a hadamard before measuring so measuring against the X basis and can use measuring from before
                        measure_any(code, i, 0) # OR NOT< only ever measure against 0 or 1 doesnt make a difference #Need to make this X basis, measuring either |+> or |->, but giving output of 1 or 0, handled in the hadamrd gate? 
        measurement.append(Result)
        return measurement
        
# Step 5 ---------------------------------------------------------------------------------------------------------------------------------------------------------------        
        
def B_Key(n, A_Bases, B_Bases, B_measure) 
        B_Key = []
        for i in range(n): 
                if A_Bases[i] = B_Bases[i]: 
                        Compared_Bits.append[measurement[i]]
                else: 
                        pass
        return B_Key
        
def A_Key(n, A_Bases, B_Bases, A_Bits) 
        A_Key = []
        for i in range(n): 
                if A_Bases[i] = B_Bases[i]: 
                        Compared_Bits.append[A_Bits[i]]
                else: 
                        pass
        return A_Key        

# Step 6 ---------------------------------------------------------------------------------------------------------------------------------------------------------------
 
 def Secret_Key(n, A_Key, B_Key) 
         sample_A = [] 
         sample_B = []
         sample_number = np.round(n*0.1) # We want roughly 10% as a sample size? 
         j = np.random.randint(sample_number, size=n)
         for k in range(j): 
               sample_A.append(A_Bits[k])
               sample_B.append(B_Bits[k])
               
               
                
                
                
        
