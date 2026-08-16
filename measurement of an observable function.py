"""
 Measurement of an observable of a density matrix
rho->density matrix,obs->observable"""

from re import I

import qutip as qt
from qutip import Qobj
import matplotlib.pyplot as plt
from qutip.measurement import measure, measurement_statistics





def expectation_value(rho: Qobj, obs: Qobj):
    """Calculates the expectation value of an observable for a given quantum state.

    Args:
        rho (Qobj): The density matrix or ket vector representing the quantum state.
        obs (Qobj): The observable operator to be measured.

    Returns:
        The expectation value of the observable.
    """
    if rho.isket:
        rho= qt.ket2dm(rho)

    return (rho * obs).tr()

def eigenvalues_vectors(obs: Qobj):
    """Calculates the eigenvalues and eigenvectors of an observable.

    Args:
        obs (Qobj): The observable operator.

    Returns:
        eigenvalues and an array of corresponding eigenvector Qobjs.
    """
    eigvals, eigvecs = obs.eigenstates()
    return eigvals, eigvecs

def measurement_probabilities(rho: Qobj, obs: Qobj):
    """Calculates the eigenvalues and their corresponding measurement probabilities using the Born rule.

    Args:
        rho (Qobj): The density matrix or ket vector representing the quantum state.
        obs (Qobj): The observable operator to be measured.

    Returns:
         the possible eigenvalues and their measurement probabilities
    """
    if rho.isket:
        rho= qt.ket2dm(rho)
    eigvals, eigvecs = eigenvalues_vectors(obs)
    probabilities = []
    
    for vec in eigvecs:
        proj = qt.ket2dm(vec) #Πi=|><|
        prob = (rho * proj).tr().real #born rule pi=Tr(ρΠi)
        probabilities.append(prob)

    return eigvals, probabilities

def plot_eigenvalue_histogram(eigvals, probabilities):
    """Plots a histogram showing the probabilities of measuring each eigenvalue.
    
        Args:
            eigvals: The possible eigenvalues of the observable.
            probabilities (list): The corresponding measurement probabilities for each eigenvalue.
    
        Returns:
            displays a matplotlib plot 
        """
    plt.bar([str(ev) for ev in eigvals], probabilities)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Probability")
    plt.title("Eigenvalue Probabilities")
    plt.show()

def multiple_eigenval_measurments(state,observable,times:int):
    """Simulates multiple measurements of an observable on a quantum state.

    Args:
        state (Qobj): The quantum state to be measured.
        observable (Qobj): The observable operator.
        times (int): The number of measurement simulations to perform.

    Returns:
        dict: A dictionary mapping measured eigenvalues to the number of times they were observed.
    """
    results={}
    for _ in range(times):
        eigenval,eigenvec=measure(state,observable)
        if eigenval not in results:
            results[eigenval] = 0
        results[eigenval]+=1
    return results

def multiple_eigenval_histogram(state,observable,times:int):
    """Plots a histogram of multiple simulated measurement results .

    Args:
        state (Qobj): The quantum state to be measured.
        observable (Qobj): The observable operator.
        times (int): The number of measurement simulations to perform.

    Returns:
         displays a matplotlib plot
    """
    results=multiple_eigenval_measurments(state,observable,times)
    plt.bar([str(key) for key in results.keys()], list(results.values()))
    plt.xlabel("Eigenvalues")
    plt.ylabel("times measured")
    plt.title("Eigenvalue measurements")
    plt.show()

if __name__ == "__main__": #oxi

    # print(multiple_eigenval_measurments(qt.rand_ket(2),qt.sigmax(),400))
    # print(multiple_eigenval_histogram(qt.basis(2,1),qt.sigmax(),400))
    # print(measure(qt.basis(2,1),qt.sigmax()))
    # eigenvalues, eigenstates, probabilities = measurement_statistics(qt.basis(2,1),qt.sigmax())
    # print(eigenvalues, eigenstates, probabilities)
    #measure first
    state00=qt.tensor(qt.basis(2,1),qt.basis(2,1))
    print(measure(state00,qt.tensor(qt.sigmax(),qt.qeye(2))))
















# #testsprint
#     #rho= qt.ket2dm(random_state(4))
#     rho=qt.tensor(qt.ket2dm(random_state(2)),qt.ket2dm(random_state(2)))
# #    # x=tensor_product(rho)
# #     # rho = (
# #     #     0.7 * qt.ket2dm(random_state(2))
# #     #     + 0.3 * qt.ket2dm(random_state(2))
# #     # )
    
#     obs = qt.tensor(qt.sigmax(),qt.qeye(2))
#     #print(qt.qeye(2))
# #     #obs= qt.rand_herm(4)
#    # print("expectation value:" , expectation_value(rho, obs))
#     print(rho * obs)
# eigvals, probs = measurement_probabilities(rho, obs)

# #     print("eigenvalues:",eigvals)
# #     print("probabilities",probs)

# plot_eigenvalue_histogram(eigvals, probs)
# print(obs)
#    # print(rho)