
import numpy as np
import qutip as qt
from qutip import Qobj

def purity(rho:Qobj) -> complex:
    """Calculates the purity of a quantum state.

    Args:
        rho (Qobj): The density matrix representing the quantum state.

    Returns:
        complex: The calculated purity value
    """
    return (rho * rho).tr()

def is_pure_purity(purity:float) -> bool:
    """Determines if a quantum state is pure based on its calculated purity value.

    Args:
        purity (float): The purity value of the state

    Returns:
        bool: True if the state is pure, False if it is mixed.
    """
    if np.isclose(purity, 1):
        return True
    else:
        return False
#crit with entropy

def is_pure_vonneumann(rho:Qobj):
    """Determines if a quantum state is pure by calculating its von Neumann entropy.
    Args:
        rho (Qobj): The density matrix representing the quantum state.

    Returns:
        bool: True if the state is pure (entropy is approximately 0), False if it is mixed.
    """
    S = qt.entropy_vn(rho, base=2)
    if np.isclose(S, 0):
       return True
    elif S>0:
       return False

# def purity_criterion(purity:float):

#     if np.isclose(purity, 1):

#         print('pure state')

#     elif np.isclose(purity, 1/2):

#         print('maximally mixed state')

#     else:

#         print('mixed state')

if __name__ == "__main__": #oxi


    zero = qt.basis(2, 0)

    rho_pure = qt.ket2dm(zero)

    pur = purity(rho_pure)

    print(pur)

    print(is_pure_purity(pur))

    # print(purity_criterion(pur))