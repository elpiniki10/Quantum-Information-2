# print("test2")
# import qutip as qt
# from qutip import Qobj
# import qiskit as qk
# import numpy as np
# from rustworkx import tensor_product
# import sympy as sp
# from qiskit.quantum_info import Statevector

# import tensor_product_function

# s= np.sqrt(1/3)*Statevector([0,1,1,0,1,0,0,0]) #
# #print(s.data)
# print(sp.Matrix(s.data))
# c= s.data.reshape(2,4)
# print(sp.Matrix(c))
# u,a,vh = np.linalg.svd(c, full_matrices= False) #vh->v transpose singular value decomposition #full matrices ignores 0s
# print(u)
# print(sp.Matrix(np.round(np.transpose(vh),10)))
# print(sp.Matrix(a))#gives λ0 and λ1
# zero = qt.basis(2,0) 
# one = qt.basis(2, 1)
# psi=(np.sqrt(1/2))*(qt.tensor(zero,one)+qt.tensor(one,one))
# #print(psi)
# c=psi.reshape(1,2)
# print(c)

# """
# def schidt(psi):
#     c=psi.reshape(dimA,dimB)
#     u,a,vh = np.linalg.svd(c, full_matrices= False)
#     return u,a,vh

# print(schidt(s))"""
import qutip as qt
import matplotlib.pyplot as plt
from qutip.measurement import measure, measurement_statistics
up = qt.basis(2, 0)

down = qt.basis(2, 1)
spin_z = qt.sigmaz()

spin_x = qt.sigmax()
measure(up, spin_z) == (1.0, up)

measure(down, spin_z) == (-1.0, down)
print(measure(up, spin_x))
left = (up - down).unit()

right = (up + down).unit()
results = {1.0: 0, -1.0: 0}  # 1 and -1 are the possible outcomes
for _ in range(1000):
   value, new_state = measure(up, spin_x)
   results[round(value)] += 1
print(results)
eigenvalues, eigenstates, probabilities = measurement_statistics(up, spin_x)
# print(eigenvalues, eigenstates, probabilities)
def plot_eigenvalue_histogram(eigvals, probabilities):# histogram of eigenvalues and probabilities
    plt.bar([str(ev) for ev in eigvals], probabilities)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.title("Eigenvalue Probabilities")
    plt.show()
plot_eigenvalue_histogram(eigenvalues,probabilities)