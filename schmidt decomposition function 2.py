# import numpy as np
# import qutip as qt
# from qiskit.quantum_info import Statevector, schmidt_decomposition

# """TESTS"""
# def auto_schmidt_decomposition(state):
#     """_summary: 1) checks if state is Qobj or Statevector(qk) 2) checks if its bipartite 2)checks if normalised ,if not it normalises 3)finds schmidt decomposition via Singular value decomposition

#     Args:
#         state (_type_): bipartite state either qt Qobj or qiskit state vector

#     Returns:
#         _type_: _description_
#     """
#     # 1)SYSTEM CHECK
#     if isinstance(state, qt.Qobj):
#         if not state.isket:
#             raise ValueError("QuTiP object must be a pure state (Ket).")
#         if len(state.dims[0]) != 2:
#             raise ValueError(f"State is not bipartite! Found {len(state.dims[0])} subsystems.")
#         dim_A, dim_B = state.dims[0]
#         vec = state.full()

#     elif isinstance(state, Statevector):
#         if len(state.dims()) != 2:
#             raise ValueError(f"State is not bipartite! Found {len(state.dims())} subsystems.")
#         # Qiskit stores dimensions right-to-left: (dim_subsystem_0, dim_subsystem_1)
#         dim_B, dim_A = state.dims()
#         vec = state.data
#     else:
#         raise TypeError("Unsupported state type. Use Qiskit Statevector, or QuTiP Qobj.")

#     # 2) NORMALIZATION CHECK
#     norm = np.linalg.norm(vec)
#     if not np.isclose(norm, 1.0, atol=1e-6):
#         vec = vec / norm

#     # 3) SCHMIDT DECOMPOSITION, SVD
#     state_matrix = vec.reshape((dim_A, dim_B))
#     U, S, Vh = np.linalg.svd(state_matrix, full_matrices=False)
    
#     return S, U, Vh, (dim_A, dim_B)

# def schmidt_decomposition(psi: qt.Qobj):
#     """ 
#     """

#     """
#     Schmidt decomposition for a bipartite pure state in QuTiP.

#     Parameters
#     ----------
#     psi : qutip.Qobj
#         A pure-state ket with dims like [[dA, dB], [1, 1]].
#     tol : float
#         Numerical cutoff for tiny singular values.

#     Returns
#     -------
#     coeffs : np.ndarray
#         Schmidt coefficients.
#     A_vecs : list[qt.Qobj]
#         Schmidt kets for subsystem A.
#     B_vecs : list[qt.Qobj]
#         Schmidt kets for subsystem B.
#     """
#     if not psi.isket:
#         raise ValueError("psi must be a ket (pure state).")

#     dims = psi.dims[0]
#     if len(dims) != 2:
#         raise ValueError(
#             "Schmidt decomposition requiress a bipartite state with dims like [[dA, dB], [1, 1]]."
#         )

#     dA, dB = dims
#     vec = np.asarray(psi.full(), dtype=complex).reshape(dA, dB)

#     U, S, Vh = np.linalg.svd(vec, full_matrices=False)

#     coeffs = S[keep]
#     A_vecs = [qt.Qobj(U[:, k], dims=[[dA], [1]]) for k in range(len(coeffs))]
#     B_vecs = [qt.Qobj(Vh[k, :].conj().T, dims=[[dB], [1]]) for k in range(len(coeffs))]

#     return coeffs, A_vecs, B_vecs


# def schmidt_qiskit(psi, qargs):
#     return schmidt_decomposition(psi, qargs)


# # Example: Bell state


# # QuTiP
# bell_qutip = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) +
#               qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()

# coeffs, A_vecs, B_vecs = schmidt_qutip(bell_qutip)
# print("QuTiP coefficients:", coeffs)
# print("A vectors:", A_vecs)
# print("B vectors:", B_vecs)

# # Qiskit
# bell_qiskit = Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
# terms = schmidt_qiskit(bell_qiskit, [1])   # subsystem B = qubit 1

# print("\nQiskit Schmidt terms:")
# for s, u, v in terms:
#     print("s =", s)
#     print("u =", u)
#     print("v =", v)
#     print()

import numpy as np
import qutip as qt
from qiskit.quantum_info import Statevector
from qutip import Qobj
from rustworkx import tensor_product

def qt_schmidt_decomposition(state:Qobj):
   
    # 1)SYSTEM CHECK
    if not state.isket:
        raise ValueError("QuTiP object must be a pure state (Ket).")
    if len(state.dims[0]) != 2:
        raise ValueError(f"State is not bipartite found {len(state.dims[0])} subsystems.")
    dim_A, dim_B = state.dims[0]
    # 2) NORMALIZATION CHECK
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0, atol=1e-6):
        state = state / norm

    # 3) SCHMIDT DECOMPOSITION, SVD
    state_matrix = state.reshape((dim_A, dim_B))
    schmidt_basisA,schmidt_coeffs,schmidt_basisB  = np.linalg.svd(state_matrix, full_matrices=False)
    # metatroph numpy -> qt giati ksekinhsa me numpy kai allaksa sthn mesh se qt
    schmidt_basisA = Qobj(schmidt_basisA)
    schmidt_basisB = Qobj(schmidt_basisB)
    return schmidt_basisA,schmidt_coeffs,schmidt_basisB, (dim_A, dim_B)


# def is_entangled(state):
#     u, s, v, d = qk_schmidt_decomposition(state)
#     schmidt_rank = np.sum(s)
#     if schmidt_rank == 1:#seperable 
#         return False
#     else:#entangled
#         return True
"""#tests
# unnormalized Bell state |00> + |11> (missing 1/sqrt(2))
bell_state = Statevector([1, 0, 0, 1]) 
u,s,vh,d=qk_schmidt_decomposition(bell_state)
x= qt.tensor(u,vh)
#print(u,s,vh,d)
#reconstructed_state = (u @ np.diag(s) @ vh).flatten()#na balw pano /ksanadw
#print("schm deco", reconstructed_state)
#is_entangled(bell_state)
#print(is_entangled(bell_state))"""

#A 3x3 dim system[010],[001]
A = qt.basis(3, 1)
B = qt.basis(3, 2)
qstate = qt.tensor(A,B)
print(qstate.dims)
# dim_A, dim_B = state.dims[0]
# state_matrix = qstate.full().reshape(dim_A, dim_B)
# s,u,vh,d=qt_schmidt_decomposition(qstate)
# print(qt.tensor(u,vh))
# print(state_matrix)
#print(s,u,vh,d)
#|000> |111>
#tripartite_state = Statevector([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
#s,u,vh,d=qk_schmidt_decomposition(tripartite_state)
#print(s)

# #example 9.4 maneti
# mstate = Statevector([0, 1, 0, 1])
# s,u,vh,d=qt_schmidt_decomposition(mstate)
# #is_entangled(state)

# print(is_entangled(state))
# reconstructed_state = (u @ np.diag(s) @ vh).flatten()#na balw pano /ksanadw
# print("schm deco", reconstructed_state)


#qiskit schmidt
def qk_schmidt_decomposition(state:Statevector):
    """_summary:computes schmidt decomposition via Singular value decomposition |ψ>=Σsk|ek>@|hk>
    Steps:
    1) Checks if state is bipartite
    2) Checks normalization and normalizes
    3) Reshapes the state vector into a bipartite matrix
    4) Applies SVD to obtain the Schmidt decomposition
    Args:
        state (_type_): bipartite state qk Statevector 

    Returns:schmidt_coeffsSchmidt coefficients sk (singular values),schmidt_basis_A (left eigenvectors for subsystem A |ek>)
    ,schmidt_basis_B (right eigenvectors for subsystem B|hk>),dims(Dimensions of the bipartite subsystems)
    
    """
    # 1)SYSTEM CHECK
    if len(state.dims()) != 2:
        raise ValueError(f"State is not bipartite! Found {len(state.dims())} subsystems.") # Qiskit stores dimensions right-to-left: (dima, dimb)
    dim_B, dim_A = state.dims()
    vec = state.data
    # 2) NORMALIZATION CHECK
    norm = np.linalg.norm(vec)
    if not np.isclose(norm, 1.0, atol=1e-6):
        vec = vec / norm
    # 3) SCHMIDT DECOMPOSITION, SVD
    state_matrix = vec.reshape((dim_A, dim_B))
    schmidt_basisA,schmidt_coeffs,schmidt_basisB = np.linalg.svd(state_matrix, full_matrices=False) #c = uav
    return schmidt_basisA,schmidt_coeffs,schmidt_basisB ,(dim_A, dim_B)
"""#tests
# unnormalized Bell state |00> + |11> (missing 1/sqrt(2))
bell_state = Statevector([1, 0, 0, 1]) 
u,s,vh,d=qk_schmidt_decomposition(bell_state)
x= qt.tensor(u,vh)
#print(u,s,vh,d)
#reconstructed_state = (u @ np.diag(s) @ vh).flatten()#na balw pano /ksanadw
#print("schm deco", reconstructed_state)
#is_entangled(bell_state)
#print(is_entangled(bell_state))"""
#print(s,u,vh,d)
#|000> |111>
#tripartite_state = Statevector([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
#s,u,vh,d=qk_schmidt_decomposition(tripartite_state)
#print(s)

# #example 9.4 maneti
# mstate = Statevector([0, 1, 0, 1])
# s,u,vh,d=qt_schmidt_decomposition(mstate)
# #is_entangled(state)

# print(is_entangled(state))
# reconstructed_state = (u @ np.diag(s) @ vh).flatten()#na balw pano /ksanadw
# print("schm deco", reconstructed_state)
#tests
# unnormalized Bell state |00> + |11> (missing 1/sqrt(2))
bell_state = Statevector([1, 0, 0, 1]) 
u,s,vh,d=qk_schmidt_decomposition(bell_state)
x= qt.tensor(u,vh)
#print(u,s,vh,d)
#reconstructed_state = (u @ np.diag(s) @ vh).flatten()#na balw pano /ksanadw
#print("schm deco", reconstructed_state)
#is_entangled(bell_state)
#print(is_entangled(bell_state))