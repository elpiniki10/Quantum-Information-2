import qutip as qt
import numpy as np
from qutip import Qobj
def is_ent_witness(rho:Qobj,W:Qobj)->bool:
    """Entanglment criterion usint ent witness
    if tr(rho*W)<0 state is entangled.Else no conclusions can bemade
    Args:
        rho (Qobj): density matrix
        W (Qobj): entanglement witness

    Returns:
        bool: entanglement confirmation
    """
    a=rho*W

    if np.real(a.tr()) <1e-10:
        return True
if __name__ == "__main__": 
    rho=qt.ket2dm(qt.bell_state("00"))
    w_SWAP=qt.swap(2,2)#2 qbts of 2 dims each
    print(is_ent_witness(rho,w_SWAP))
    zero=qt.basis(2,0)
    one=qt.basis(2,1)
    print(is_ent_witness(rho,w_SWAP))
