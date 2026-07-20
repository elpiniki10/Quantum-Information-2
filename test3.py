import numpy as np
import qutip as qt
import qiskit as qk


def state_0():
    return qt.basis(2,0)
def state_1():
    return qt.basis(2, 1)
def state_plus():
    return (qt.basis(2, 0) + qt.basis(2, 1)).unit()
def state_00():
    return qt.tensor(qt.basis(2,0),qt.basis(2,0))
def state_11():
    return qt.tensor(qt.basis(2, 1),qt.basis(2, 1))



import qutip as qt
import matplotlib.pyplot as plt
from qutip.measurement import measure, measurement_statistics
up = qt.basis(2, 0)

down = qt.basis(2, 1)
spin_z = qt.sigmaz()

spin_x = qt.sigmax()
measure(up, spin_z) == (1.0, up) #returns eigenvalue,eigenvector

measure(down, spin_z) == (-1.0, down)
#print(measure(up, spin_x))
left = (up - down).unit()

right = (up + down).unit()



# results = {1.0: 0, -1.0: 0}  # 1 and -1 are the possible outcomes
# for _ in range(1000):
#    value, new_state = measure(up, spin_x)
#    results[round(value)] += 1
# print(results)



# eigenvalues, eigenstates, probabilities = measurement_statistics(up, spin_x)

def multiple_eigenval_measurments(state,observable,times:int):
    results={}
    for _ in range(times):
        eigenval,eigenvec=measure(state,observable)
        if eigenval not in results:
            results[eigenval] = 0
        results[eigenval]+=1
    return results
def multiple_eigenval_histogram(state,observable,times:int):
    results=multiple_eigenval_measurments(state,observable,times)
    plt.bar([str(key) for key in results.keys()], list(results.values()))
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.title("Eigenvalue Probabilities")
    plt.show()
print(multiple_eigenval_histogram(up, spin_x,500))
  
# # print(eigenvalues, eigenstates, probabilities)
# def plot_eigenvalue_histogram(eigvals, probabilities):# histogram of eigenvalues and probabilities
#     plt.bar([str(ev) for ev in eigvals], probabilities)
#     plt.xlabel("Eigenvalues")
#     plt.ylabel("Probability")
#     plt.title("Eigenvalue Probabilities")
#     plt.show()
# plot_eigenvalue_histogram(eigenvalues,probabilities)