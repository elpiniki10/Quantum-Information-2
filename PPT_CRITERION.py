import qutip as qt
from qutip import Qobj

def PPT_crit_sufficient(rho:Qobj)->bool:
    """Applies the Positive Partial Transpose (PPT) criterion to definitively check for entanglement.
    This provides a necessary and sufficient condition for a mixed state of bipartite 
    subsystems to be entangled. If the partially transposed density matrix has negative 
    eigenvalues, the state is entangled. This strict necessity and sufficiency is valid 
    only for Hilbert space dimensions 2x2, 2x3, and 3x2.

    Args:
        rho (Qobj): The density matrix or ket vector of the quantum state to check.

    Returns:
        bool: True if the state is entangled, False if it is separable.
        
    Raises:
        ValueError: If the subsystem dimensions are not [2,2], [2,3], or [3,2].
    """
    if rho.isket:
        rho= qt.ket2dm(rho)
    dims=rho.dims[0] #rho is a density matrix bc ppt is for mixed therefore its square and row dims=col represent hilbert dims 
    if dims in [[2,2],[2,3],[3,2]]:
        ptrans_rho=qt.partial_transpose(rho,mask=[0,1]) #mask -> Apply identity to the first subsystem ansd partially transpose the second
        eigenvals=  ptrans_rho.eigenenergies() #gives eigenvalues in ascending order therefore negative ones always go first so the for loop is safe
        for λ in eigenvals:
            if  λ <-1e-8:
                return True
        else: 
            return False
    else:
        raise ValueError("PPT criterion not necessary and sufficient for  this Hilbert space.")
    
def PPT_crit(rho:Qobj)->bool:
    """Positive partial transpose criterion.Sufficient condition
    Applies the Positive Partial Transpose (PPT) criterion.Sufficient condition

    Args:
        rho (Qobj): The density matrix or ket vector of the quantum state to check.

    Returns:
        bool: True if the state is entangled (negative eigenvalue detected), False otherwise.
    """
    if rho.isket:
        rho= qt.ket2dm(rho)
    ptrans_rho=qt.partial_transpose(rho,mask=[0,1]) #mask -> Apply identity to the first subsystem ansd partially transpose the second
    eigenvals=  ptrans_rho.eigenenergies() #gives eigenvalues in ascending order therefore negative ones always go first so the for loop is safe
    for λ in eigenvals:
        if  λ <-1e-8:
            return True
    else: 
        return False
    
# if __name__ == "__main__":
#     state_A = qt.rand_ket(4)
#     state_B = qt.rand_ket(2)
#     rho = qt.tensor(state_A, state_B)
#     #print(rho)
#     print("is ent?",PPT_crit_sufficient(rho))
#     print("is ent?",PPT_crit(rho))
#     #bell example
#     bell_00= qt.bell_state('00')
#     print("is ent?",PPT_crit_sufficient(bell_00))
