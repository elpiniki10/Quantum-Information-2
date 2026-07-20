"""
 Partial trace of a composite density matrix
"""

import qutip as qt
from qutip import Qobj
def partial_traceA(A:Qobj,B:Qobj)->Qobj:
    rho_AB = qt.tensor(A, B)
    rho_A = rho_AB.ptrace(0)
    return rho_A

def partial_traceB(A:Qobj,B:Qobj)->Qobj:
    """_summary_

    Args:
        A (Qobj): _description_
        B (Qobj): _description_

    Returns:
        Qobj: _description_
    """
    rho_AB = qt.tensor(A, B)
    rho_B = rho_AB.ptrace(1)
    return rho_B
if __name__ == "__main__": #oxi
    #TESTS
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    rho1 = qt.ket2dm(zero)
    rho2 = 0.5 * qt.ket2dm(zero) + 0.5 * qt.ket2dm(one)
    print(partial_traceB(rho1, rho2))
    print(partial_traceA(rho1, rho2))