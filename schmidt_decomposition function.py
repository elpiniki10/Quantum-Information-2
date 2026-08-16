
import numpy as np
import qutip as qt
from qutip import Qobj




def qt_schmidt_decomposition(state:Qobj):
    """_summary:computes schmidt decomposition via Singular value decomposition |ψ>=Σsk|ek>@|hk>
    Steps:
    1) Checks if state is bipartite
    2) Checks normalization and normalizes
    3) Reshapes the state vector into a bipartite matrix
    4) Applies SVD to obtain the Schmidt decomposition
    Args:
        state (_type_): bipartite state qt Qobj 

    Returns:schmidt_coeffs(Schmidt coefficients sk,singular values),schmidt_basis_A (left eigenvectors for subsystem A |ek>)
    ,schmidt_basis_B(right eigenvectors for subsystem B|hk>),dims(Dimensions of the bipartite subsystems)
      
    """
    # 1)SYSTEM CHECK
    if not state.isket:#schmidt is for pure
        raise ValueError("QuTiP object must be a pure state (Ket).")
    if len(state.dims[0]) != 2: #checks if system is bipartite.len(list) counts how many objects in a list
        raise ValueError(f"State is not bipartite")
    dim_A, dim_B = state.dims[0]#state.dims=[[list of row diamentions],[list of column dims]] for kets list of column dims always 1 so state.dims[0] gives [a,b] where a,b row dims only (if i had index 1 id get only columns)
    state=state.unit() #normalises
    vec = state.full()#turns it into numpy array to use np funcs later
    # 3) SCHMIDT DECOMPOSITION, SVD
    state_matrix = vec.reshape((dim_A, dim_B))# Writes the ket like ΣΣcij|i>@|j>             (since pure state is ket every obj of the martix is on one column. reshapes into 2d coef matrx, changes dims-> matrix=(new rows,new columns))
    schmidt_basisA,schmidt_coeffs,schmidt_basisB  = np.linalg.svd(state_matrix, full_matrices=False)
   


    schmidt_dec = {} #dictionary
    for k, sk in enumerate(schmidt_coeffs): #k ->index, sk->value of coef (ennumerate returns pairs in the form (index, element))
        ek = schmidt_basisA[:, k] #[:,...]gives all rows and only k column U
        hk = schmidt_basisB[k, :]#its v CONJ so only k row all cols
        schmidt_dec[float(sk)] = ( Qobj(np.real_if_close(ek)),Qobj(np.real_if_close(hk)))#If input is complex with all imaginary parts close to zero, return real parts.
        
     # metatropi numpy->qt
    schmidt_basisA = Qobj(schmidt_basisA)
    schmidt_basisB = Qobj(schmidt_basisB)
    return schmidt_basisA,schmidt_coeffs,schmidt_basisB,schmidt_dec, (dim_A, dim_B)

def is_entangled(state):
    """Determines if a pure bipartite state is entangled by evaluating its Schmidt rank.

    Args:
        state (Qobj): The pure bipartite quantum state to check.

    Returns:
        bool: True if the state is entangled (Schmidt rank > 1), False if it is separable (Schmidt rank = 1).
    """
    u, s, v,S, d = qt_schmidt_decomposition(state)
    schmidt_rank = np.count_nonzero(s>1e-10)#counts non zero schmidt coefficients
    if schmidt_rank == 1:#seperable 
        return False
    else:#entangled
        return True
def confirm_schmidt(psi):
        """Verifies that the calculated Schmidt decomposition accurately reconstructs the original quantum state.

    Args:
        psi (Qobj): The original pure bipartite state (ket) to be decomposed and reconstructed.

    Returns:
        bool: True if the reconstructed state matches the original state, False otherwise.
    """
        u,s,vh,S,d=qt_schmidt_decomposition(psi)
        reconstructed_state=0
        for sk in S:
            u,v=S[sk]
            reconstructed_state+=sk*qt.tensor(u,v)
        if np.allclose(psi.full(),reconstructed_state.full()):#allclose checks if all elements of 2 np arrays are close (full turns into np)
            return True
        else:
            return False
if __name__ == "__main__":
    # # A 3x3 dim system[010],[001]
    # A = qt.basis(3, 1)
    # B = qt.basis(3, 2)
    # qstate = qt.tensor(A,B)
    # u,s,vh,S,d=qt_schmidt_decomposition(qstate)
    # print(is_entangled(qstate))
    # print(s,S)

    # bell00= qt.bell_state('00')
    # m,r,h,g,d=qt_schmidt_decomposition(bell00)
    # print(is_entangled(bell00))
    # print(r,g)

    zero = qt.basis(2, 0)  # |0>
    one  = qt.basis(2, 1)  # |1>
    psi00 = qt.tensor(zero, zero)
    psi01 = qt.tensor(zero, one)   
    psi10 = qt.tensor(one, zero)   
    psi11 = qt.tensor(one, one)    
    # #ex 1
    u,s,vh,S,d=qt_schmidt_decomposition(psi11)
    print(S)
    
    
    print(confirm_schmidt(psi11))            
    
    # # #example schmid 2
    # qqstate = 0.5 * psi00 - 0.5 * psi01 - 0.5 * psi10 + 0.5 * psi11
    # u,s,vh,S,d=qt_schmidt_decomposition(qqstate)
    # print(s,S)
    # #9.4 maneti
    # mstate=(1/np.sqrt(2))*(psi01+psi11)
    # u, s, vh, S, d = qt_schmidt_decomposition(mstate)

    # print(S, is_entangled(mstate))
    
    # x=random_state_function.random_state(6)
    # y=random_state_function.random_state(3)
    # e=qt.tensor(x,y)
    # u,s,vh,S,d=qt_schmidt_decomposition(e)
    # print(is_entangled(e),S)
