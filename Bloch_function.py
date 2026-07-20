"""
Block vector ->blochvector()
blochvectorlength-> lenght of bloch vector
bloch_criterion-> criterion of pure or mixed state based on bloch vec length
"""

import qutip as qt
import numpy as np
from qutip import Qobj
from qutip import Bloch

def blochvector(rho:Qobj,x:Qobj,y:Qobj,z:Qobj) -> float: 
    """_summary_

    Args:
        rho (Qobj): _description_
        x (Qobj): _description_
        y (Qobj): _description_
        z (Qobj): _description_

    Returns:
        float: _description_
    """
    rx = (rho * x).tr().real
    ry = (rho * y).tr().real
    rz = (rho * z).tr().real
    r = np.array([rx, ry, rz])
    return r
def blochvectorlength(r):
    length= np.linalg.norm(r)
    return length

def is_pure_bloch(length:float)->bool:
    """
    """
    if np.isclose(length, 1):
        return True
    else:
        return False
def bloch_sphere(r):
    b=Bloch()
    b.add_vectors(r)
    b.save("bloch.png")
    b.show()
if __name__ == "__main__": #oxi

    #test
    psi = (qt.basis(2,0) + qt.basis(2,1)).unit()
    rho = qt.ket2dm(psi)
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()
    r=blochvector(rho,sx,sy,sz)
    print(r)
    l=blochvectorlength(r)
    print(l)
    print(is_pure_bloch (l))
    bloch_sphere(r)
