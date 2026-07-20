import qutip as qt
import numpy as np
from qutip import Qobj

def werner()->Qobj:
    """generates werner state based on probability given by user input 

    Raises:
        ValueError: probability 0<p<1

    Returns:
        Qobj: werner state
    """
    zero=qt.basis(2,0)
    one=qt.basis(2,1)
    p=float(input("probability:"))
    if p>1 or p<0:
        raise ValueError("probability 0<p<1")
    # rhowerner=p*(qt.ket2dm(qt.tensor(zero,zero))+qt.tensor(qt.tensor(one,one),qt.tensor(zero,zero).dag()))
    werner=p*qt.ket2dm(qt.bell_state('00'))+(1-p)*qt.tensor((qt.qeye(2)/2),(qt.qeye(2)/2))
    return werner
if __name__ == "__main__":# test
    print(werner())
