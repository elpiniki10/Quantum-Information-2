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
    """Calculates the Bloch vector coordinates for a given quantum state.

    Args:
        rho (Qobj): The density matrix representing the state of the qubit.
        x (Qobj): The Pauli-X operator.
        y (Qobj): The Pauli-Y operator.
        z (Qobj): The Pauli-Z operator.

    Returns:
        float: A 3-element numpy array containing the (rx, ry, rz) coordinates of the Bloch vecto
    """
    rx = (rho * x).tr().real
    ry = (rho * y).tr().real
    rz = (rho * z).tr().real
    r = np.array([rx, ry, rz])
    return r

def blochvectorlength(r):
    """Calculates the Euclidean norm (length) of a given Bloch vector.

    Args:
        r : A 3-element array representing the (rx, ry, rz) coordinates of the Bloch vector.

    Returns:
        float: The length of the Bloch vector.
    """
    length= np.linalg.norm(r)
    return length

def is_pure_bloch(length:float)->bool:
    """
    Determines if a quantum state is pure based on its Bloch vector length.

    Args:
        length (float): The calculated length of the Bloch vector.

    Returns:
        bool: True if the state is pure ,false if it is mixed.
    """
    if np.isclose(length, 1):
        return True
    else:
        return False
def bloch_sphere(r):
    """Visualizes the Bloch vector on a 3D Bloch sphere, saves it, and displays the plot.

    Args:
        r : A 3-element array representing the (rx, ry, rz) coordinates of the Bloch vector.

    Returns:
        None: saves "bloch.png"
    """
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
