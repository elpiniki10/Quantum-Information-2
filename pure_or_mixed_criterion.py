"""
criterion of whether a state is pure or mixed
"""
#crit with purity
import numpy as np
import qutip as qt
from qutip import Qobj

def purity(rho:Qobj) -> complex:
    return (rho * rho).tr()

def is_pure_purity(purity:float) -> bool:
    if np.isclose(purity, 1):
        return True
    else:
        return False
#crit with entropy

def is_pure_vonneumann(rho:Qobj):
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