from math import sqrt
import qutip as qt
from qutip import Qobj

def PCC(Sab:Qobj,A:Qobj,B:Qobj):
    """Calculates the Pearson Correlation coefficient (eq 2)

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (Qobj): observable,hermitian matrix a e Ha
        B (Qobj): observable,hermitian matrix b e Hb

    Returns:
        float: Correlation of AB in range[-1,1] (0 if a variance vanishes, where the PCC is undefined)
    """
    #tensors
    a=qt.tensor(A,B)
    b=qt.tensor(A,qt.qeye(B.shape[0]))
    c=qt.tensor(qt.qeye(A.shape[0]),B)

    #expectation values
    Ea=(Sab*b).tr()
    Eb=(Sab*c).tr()
    Eab=(Sab*a).tr()
    Covab=Eab-Ea*Eb #covariance a,b

    #variances a,b
    Va=(Sab*(b**2)).tr()-(Ea**2)

    Vb=(Sab*(c**2)).tr()-(Eb**2)

    VaVb=(Va*Vb).real
    if VaVb<=1e-12: # vanishing variance: PCC undefined
        return 0.0
    Correlation_ab=Covab/sqrt(VaVb)
    return Correlation_ab.real

def sum_PCC(Sab:Qobj,A:list[Qobj],B:list[Qobj])->float:#A1,A2,A3,B1,B2,B3
    """Σ|Cor{Ai,Bi}| over the given observables.
    With 3 observables it is the left hand side of eq 15, with 2 of eq 33 (MBM)

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 or 3 observables,hermitian matrices a e Ha

        B (list[Qobj]):list of 2 or 3 observables,hermitian matrices b e Hb

    Raises:
        ValueError: A and B must have the same number of observables (at most 3)

    Returns:
        float: Σ|Cor{Ai,Bi}|
    """
    if len(A)!=len(B) or len(A)>3:
        raise ValueError("A and B must have the same number of observables (at most 3)")
    sumPCC=0
    for a,b in zip( A ,B):
        sumPCC+=abs(PCC(Sab,a,b))
    return sumPCC

def is_ent_PCC(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:#A1,A2,A3,B1,B2,B3
    """PCC based entanglement criterion.Bipartite Sab is ent if A1,A2,A3 e Ha ,B1,B2,B3 e Hb give
    3Σ|Corab|>1 (15)

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 3 observables,hermitian matrices a e Ha

        B (list[Qobj]):list of 3 observables,hermitian matrices b e Hb

    Raises:
        ValueError: must be 3 obs

    Returns:
        bool: entanglement confirmation
    """
    if len(A)!=3 or len(B)!=3:
        raise ValueError("Works for 3 observables")
    return sum_PCC(Sab,A,B)>1

def is_ent_MBM(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """MBM conjecture.Bipartite Sab is ent if complementary observables A1,A2 e Ha ,B1,B2 e Hb satisfy
    2Σ^2|Corab|>1 (eq 33 on paper)
    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        bool: entanglement confirmation
    """
    if len(A)!=2 or len(B)!=2:
        raise ValueError("Works for 2 observables")
    return sum_PCC(Sab,A,B)>1 #do for<=1 seperable func

def is_ent_conjectureMBM(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """MBM conjecture.Bipartite Sab is ent if complementary observables A1,A2 e Ha ,B1,B2 e Hb satisfy
    2Σ|Corab|>2-4/(2+d) (eq 38)

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        bool: entanglement confirmation
    """
    if len(A)!=2 or len(B)!=2:
        raise ValueError("Works for 2 observables")
    d=Sab.dims[0][0]
    return sum_PCC(Sab,A,B)>2-4/(2+d)

def is_ent_strengthenedCHSH1(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """Strengthened CHSH inequality (eq 42). Sab is entangled if
    E{A1B1}+E{A1B2}+E{A2B1}-E{A2B2}>sqrt2,
    for complementary observables A1,A2 B1,B2 with dichotomic outcomes +-1

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha
        B (list[Qobj]): list of 2 observables,hermitian matrices b e Hb

    Raises:
        ValueError: must be 2 obs

    Returns:
        bool: entanglement confirmation
    """
    return CHSH1_value(Sab,A,B)>sqrt(2)

def CHSH1_value(Sab:Qobj,A:list[Qobj],B:list[Qobj]):
    """Left hand side of eq 42: E{A1B1}+E{A1B2}+E{A2B1}-E{A2B2}"""
    if len(A)!=2 or len(B)!=2:
        raise ValueError("Works for 2 observables")
    # expectation values
    #a1b1
    a = qt.tensor(A[0], B[0])
    Eab11 = (Sab * a).tr()
    # a1b2
    a = qt.tensor(A[0], B[1])
    Eab12 = (Sab * a).tr()
    # a2b1
    a = qt.tensor(A[1], B[0])
    Eab21 = (Sab * a).tr()
    # a2b2
    a = qt.tensor(A[1], B[1])
    Eab22 = (Sab * a).tr()
    s=Eab11+Eab12+Eab21-Eab22
    return s.real

def is_ent_strengthenedCHSH2(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """All bipartite q states Sab measured by complementary observables A1,A2 B1,B2 with dichotomic outcomes +-1
    that satisfy Cova1b1+Cova1b2+Cova2b1-Cova2b2<=sqrt2 are seperable (eq 43)

    Args:
        Sab (Qobj):density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha
        B (list[Qobj]):list of 2 observables,hermitian matrices b e Hb
    Raises:
        ValueError: must be 2 obs
    Returns:
        bool: entanglement confirmation
    """
    return CHSH2_value(Sab,A,B)>sqrt(2)

def CHSH2_value(Sab:Qobj,A:list[Qobj],B:list[Qobj]):
    """Left hand side of eq 43: Cov{A1,B1}+Cov{A1,B2}+Cov{A2,B1}-Cov{A2,B2}"""
    if len(A)!=2 or len(B)!=2:
        raise ValueError("Works for 2 observables")
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
    return s.real

def is_ent_strengthenedCHSH3(Sab:Qobj,A:list[Qobj],B:list[Qobj])->bool:
    """PCC based strengthened CHSH inequality (eq 46). Sab is entangled if
    Cor{A1,B1}+Cor{A1,B2}+Cor{A2,B1}-Cor{A2,B2}>sqrt2, for complementary observables A1,A2 B1,B2

    Args:
        Sab (Qobj): density matrix bipartite quantum state
        A (list[Qobj]): list of 2 observables,hermitian matrices a e Ha
        B (list[Qobj]): list of 2 observables,hermitian matrices b e Hb

    Raises:
        ValueError: must be 2 obs

    Returns:
        bool: entanglement confirmation
    """
    return CHSH3_value(Sab,A,B)>sqrt(2)

def CHSH3_value(Sab:Qobj,A:list[Qobj],B:list[Qobj]):
    """Left hand side of eq 46: Cor{A1,B1}+Cor{A1,B2}+Cor{A2,B1}-Cor{A2,B2}.exact for 2 qubits"""
    if len(A)!=2 or len(B)!=2:
        raise ValueError("Works for 2 observables")
    cor11=PCC(Sab,A[0],B[0])
    cor12=PCC(Sab,A[0],B[1])
    cor21=PCC(Sab,A[1],B[0])
    cor22=PCC(Sab,A[1],B[1])
    return cor11+cor12+cor21-cor22

def is_entangled_PPT(rho:Qobj):
    """Peres-Horodecki: a 2-qubit state is entangled iff its partial transpose has a negative eigenvalue"""
    return qt.partial_transpose(rho,[0,1]).eigenenergies().min() < -1e-12

if __name__ == "__main__":
    #print(PCC(qt.ket2dm(qt.bell_state('00')),qt.sigmax(),qt.sigmax())) #1
    #print(PCC(qt.ket2dm(qt.bell_state('10')),qt.sigmaz(),qt.sigmaz())) #-1
    # Pauli_A = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    # Pauli_B = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    #print(is_ent_PCC(qt.ket2dm(qt.bell_state('00')),Pauli_A,Pauli_B))
    a = [qt.sigmax(), qt.sigmay()]
    b = [qt.sigmax(), qt.sigmay()]
    print(is_ent_strengthenedCHSH3(qt.ket2dm(qt.bell_state('00')),a,b))
    print(is_ent_MBM(qt.ket2dm(qt.bell_state('00')),a,b))
    print(is_ent_conjectureMBM(qt.ket2dm(qt.bell_state('00')),a,b))
    print(is_ent_strengthenedCHSH2(qt.ket2dm(qt.bell_state('00')),a,b))
