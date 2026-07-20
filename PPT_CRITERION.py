import qutip as qt
from qutip import Qobj

def PPT_crit_sufficient(rho:Qobj)->bool:
    """Positive partial transpose criterion :provides necessery and sufficient condition for mixed state of 2 subsystems to be entangled if (1a@T)ρ<0 ->εντ
    
    only for {2,2}{2,3}{3,2}
    Args:
        rho (Qqoj): density matrix
    Returns:
        Boolean: Is entangled?
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
    """Positive partial transpose criterion :provides necessery and sufficient condition for mixed state of 2 subsystems to be entangled if (1a@T)ρ<0 ->εντ
    
    only for {2,2}{2,3}{3,2}
    Args:
        rho (Qqoj): density matrix
    Returns:
        Boolean: Is entangled?
    """
    if rho.isket:
        rho= qt.ket2dm(rho)
    ptrans_rho=qt.partial_transpose(rho,mask=[0,1]) #mask -> Apply identity to the first subsystem ansd partially transpose the second
    eigenvals=  ptrans_rho.eigenenergies() #gives eigenvalues in ascending order therefore negative ones always go first so the for loop is safe
    for λ in eigenvals:
        if  λ <-1e-8:
            return True
    else: #else is outside so even if the neg λ was second it would still decect ti 
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
