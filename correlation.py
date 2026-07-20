from math import sqrt
from statistics import correlation
import trace
import qutip as qt
import numpy as np
from qutip import Qobj
from sympy import false

import werner
def PCC(Sab:Qobj,A:Qobj,B:Qobj):
    """Calculates the Pearson Correlation coefficient

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (Qobj): observable,hermitian matrix a e Ha
        B (Qobj): observable,hermitian matrix b e Hb

    Returns:
        _type_: Correlation of AB in range[-1,1]
    """
    #tensors
    a=qt.tensor(A,B)
    b=qt.tensor(A,qt.qeye(2))
    c=qt.tensor(qt.qeye(2),B)
    
    #expectation values
    Ea=(Sab*b).tr()
    Eb=(Sab*c).tr()
    Eab=(Sab*a).tr()
    Covab=Eab-Ea*Eb #covariance a,b
    
    #variances a,b
    Va=(Sab*(b**2)).tr()-(Ea**2)
    
    Vb=(Sab*(c**2)).tr()-(Eb**2)

    Correlation_ab=Covab/(sqrt((Va*Vb).real))
    return Correlation_ab.real


def is_ent_PCC(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:#A1,A2,A3,B1,B2,B3
    """PCC based entanglement criterion.Bipartite Sab is ent if A1,A2,A3 e Ha ,B1,B2,B3 e Hb give
    3Σ|Corab|>1

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 3 observables,hermitian matrices a e Ha
        
        B (list[Qobj]):list of 3 observables,hermitian matrices b e Hb

    Raises:
        ValueError: must be 3 obs

    Returns:
        _type_: Boolean entanglement
    """
    if len(A)>3 and len(B)>3:
        raise ValueError("Works for 3 observables")
    sumPCC=0
    for a,b in zip( A ,B):
        X=PCC(Sab,a,b)
        sumPCC+=abs(X)
    if sumPCC>1:
        return True
    else:
        return False
def is_ent_MBM(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """MBM conjecture.Bipartite Sab is ent if complementary observables A1,A2 e Ha ,B1,B2 e Hb satisfy
    2Σ^2|Corab|>1
    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha  
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        _type_: Boolean ,entanglement confirmation
    """
    if len(A)>2 and len(B)>2:
        raise ValueError("Works for 2 observables")
    sumMBM=0
    for a,b in zip( A ,B):
        X=PCC(Sab,a,b)
        sumMBM+=abs(X)
    if (sumMBM)**2>1:
        return True
    else:
        return False #do for<=1 seperable func

def is_ent_conjecture(Sab:Qobj,A:list[Qobj],B:list[Qobj]):
    """MBM conjecture.Bipartite Sab is ent if complementary observables A1,A2 e Ha ,B1,B2 e Hb satisfy
    2Σ|Corab|>2-4/(2+d)

    Args:
       density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha  
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        _type_: Boolean ,entanglement confirmation
    """
    d=len(Sab.dims[0])
    if len(A)>2 and len(B)>2:
        raise ValueError("Works for 2 observables")
    sum=0
    for a,b in zip( A ,B):
        X=PCC(Sab,a,b)
        sum+=abs(X)
    if (sum)**2>2-4/(2+d):
        return True
    else:
        return False
def is_seperable(Sab:Qobj,A:list[Qobj],B:list[Qobj]):
    """All bipartite q states Sab measured by complementary observables A1,A2 B1,B2
    that satisfy Cova1b1+Cova1b2+Cova2b1-Cova2b2<=sqrt2 are seperable

    Args:
        Sab (Qobj):density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha  
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        _type_: Boolean ,seperability confirmation
    """
    #a1b1
    a = qt.tensor(A[0], B[0])
    b = qt.tensor(A[0], qt.qeye(2))
    c = qt.tensor(qt.qeye(2), B[0])

    # expectation values
    Ea = (Sab * b).tr()
    Eb = (Sab * c).tr()
    Eab = (Sab * a).tr()
    Cova1b1 = Eab - Ea * Eb
    
    # a1b2 
    a = qt.tensor(A[0], B[1])
    b = qt.tensor(A[0], qt.qeye(2))
    c = qt.tensor(qt.qeye(2), B[1])

    # expectation values
    Ea = (Sab * b).tr()
    Eb = (Sab * c).tr()
    Eab = (Sab * a).tr()
    Cova1b2 = Eab - Ea * Eb
    
    # a2b1 
    a = qt.tensor(A[1], B[0])
    b = qt.tensor(A[1], qt.qeye(2))
    c = qt.tensor(qt.qeye(2), B[0])

    # expectation values
    Ea = (Sab * b).tr()
    Eb = (Sab * c).tr()
    Eab = (Sab * a).tr()
    Cova2b1 = Eab - Ea * Eb
    
    # a2b2 
    a = qt.tensor(A[1], B[1])
    b = qt.tensor(A[1], qt.qeye(2))
    c = qt.tensor(qt.qeye(2), B[1])

    # expectation values
    Ea = (Sab * b).tr()
    Eb = (Sab * c).tr()
    Eab = (Sab * a).tr()
    Cova2b2 = Eab - Ea * Eb
    s=Cova1b1+Cova1b2+Cova2b1-Cova2b2
    if s.real<=sqrt(2):
        return True
    else:
        return False
if __name__ == "__main__":
    #print(PCC(werner.werner(),qt.sigmaz(),qt.sigmaz()))#
    #print(PCC(qt.ket2dm(qt.bell_state('00')),qt.sigmaz(),qt.sigmaz())) #1
    #print(PCC(qt.ket2dm(qt.bell_state('10')),qt.sigmaz(),qt.sigmaz())) #-1
    Pauli_A = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    Pauli_B = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    #print(is_ent_PCC(qt.ket2dm(qt.bell_state('00')),Pauli_A,Pauli_B))
    a = [qt.sigmax(), qt.sigmay()]
    b = [qt.sigmax(), qt.sigmay()]
    # print(is_ent_MBM(qt.ket2dm(qt.bell_state('00')),a,b))
    print(is_ent_conjecture(qt.ket2dm(qt.bell_state('00')),a,b))
    print(is_seperable(qt.ket2dm(qt.tensor((qt.basis(2,0)),qt.basis(2,0))),a,b))
    print(is_seperable(qt.ket2dm(qt.bell_state('00')),a,b))