import warnings
import numpy as np
import qutip as qt
from qiskit.quantum_info import Statevector
from rustworkx import tensor_product


# def _extract_vec_and_dims(state:qt.Qobj):
#     if not state.isket:
#         raise ValueError("QuTiP object must be a pure state (Ket).")
#     if len(state.dims[0]) != 2:
#         raise ValueError(f"State is not bipartite! Found {len(state.dims[0])} subsystems.")
#     dim_A, dim_B = state.dims[0]
#     return state.full().ravel(), dim_A, dim_B

#     if isinstance(state, Statevector):
#         dims = state.dims()
#         if len(dims) != 2:
#             raise ValueError(f"State is not bipartite! Found {len(dims)} subsystems.")
#         dim_B, dim_A = dims  # keep this only if it matches your intended subsystem order
#         return np.asarray(state.data).ravel(), dim_A, dim_B

#     if isinstance(state, np.ndarray):
#         if state.ndim == 2:
#             dim_A, dim_B = state.shape
#             return state.ravel(), dim_A, dim_B

#         if state.ndim == 1:
#             n = state.size
#             root = int(np.sqrt(n))
#             if root * root == n:
#                 return state, root, root
#             raise ValueError(
#                 f"Cannot auto-detect dimensions for 1D NumPy array of length {n}. "
#                 "Pass a Qiskit Statevector, QuTiP Qobj, or a 2D matrix instead."
#             )

#     raise TypeError(
#         "Unsupported state type. Use NumPy array, Qiskit Statevector, or QuTiP Qobj."
#     )


# def auto_schmidt_decomposition(state):
#     vec, dim_A, dim_B = _extract_vec_and_dims(state)

#     norm = np.linalg.norm(vec)
#     if norm == 0:
#         raise ValueError("State has zero norm.")
#     if not np.isclose(norm, 1.0, atol=1e-6):
#         warnings.warn(f"State not normalized (norm = {norm:.5f}); auto-normalizing.", RuntimeWarning)
#         vec = vec / norm

#     state_matrix = vec.reshape(dim_A, dim_B)
#     U, S, Vh = np.linalg.svd(state_matrix, full_matrices=False)

#     S = np.where(np.isclose(S, 0, atol=1e-9), 0, S)
#     return S, U, Vh, (dim_A, dim_B)

# one  = qt.basis(2, 1)
# ze=qt.basis(3,0)
# y=qt.tensor(one,ze)
# d=k
# x= y.dims
# print(x)
# # from qutip import *

# psi = tensor(basis(2, 0), basis(2, 1))
# print(psi)  # Output: (4, 1)
# print(psi.dims)   # Output: [[2, 2], [1, 1]]

import numpy as np
from numpy import pi
from qutip import Qobj, about
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import (berkeley, cnot, cphase, csign, fredkin,
                                  gate_sequence_product, globalphase, iswap,
                                  molmer_sorensen, phasegate, qrot, rx, ry, rz,
                                  snot, sqrtiswap, sqrtnot, sqrtswap, swap,
                                  swapalpha, toffoli)
from qutip_qip.transpiler import to_chain_structure

# from matplotlib import inline
# cphase(pi / 2)
# q = QubitCircuit(2, reverse_states=False)
# # q.add_gate("CSIGN", controls=[0], targets=[1])
# # q.draw()
# teleportation = QubitCircuit(
#     3, num_cbits=2, input_states=[r"\psi", "0", "0", "c0", "c1"]
# )
# teleportation.add_gate("SNOT", targets=[1])
# teleportation.add_gate("CNOT", targets=[2], controls=[1])
# teleportation.add_gate("CNOT", targets=[1], controls=[0])
# teleportation.add_gate("SNOT", targets=[0])
# teleportation.add_measurement("M0", targets=[0], classical_store=1)
# teleportation.add_measurement("M1", targets=[1], classical_store=0)
# teleportation.add_gate("X", targets=[2], classical_controls=[0])
# teleportation.add_gate("Z", targets=[2], classical_controls=[1])
# teleportation.add_measurement("M2", targets=[2], classical_store=0)
# teleportation.gates


#other tel
# psi = rand_ket(2) #to be teleported

# psi0 = tensor([psi, basis(2, 0), basis(2, 0)])

# psi1 = snot(N=3, target=1)*psi0
# psi2 = cnot(N=3, control=1, target=2)*psi1
# psi3 = cnot(N=3, control=0, target=1)*psi2
# psi4 = snot(N=3, target=0)*psi3

# confs = list(itertools.product([0, 1], repeat=2))

# Ps = []
# for m0, m1 in confs:
#     P = tensor([
#         basis(2, m0).proj(),
#         basis(2, m1).proj(),
#         qeye(2)])
#     Ps.append(P)

# psis_proj = []
# for P in Ps:
#     psi_proj = (P*psi4).unit()
#     psis_proj.append(psi_proj)


# X = rx(np.pi, N=3, target=2)
# Z = rz(np.pi, N=3, target=2)
# psis_corr = [
#     psis_proj[0],
#     X*psis_proj[1],
#     Z*psis_proj[2],
#     Z*X*psis_proj[3]
# ]
# psis_ref = []
# for m0, m1 in confs:
#     psi_ref = tensor([basis(2, m0), basis(2, m1), psi])
#     psis_ref.append(psi_ref)
# print('{0:2} {1:2} {2:8}'.format('m0', 'm1', 'fidelity'))
# for conf, psi_corr, psi_ref in zip(confs, psis_corr, psis_ref):
#     fidelity = np.round(np.abs(psi_corr.overlap(psi_ref))**2., 3)
#     m0, m1 = conf
#     print('{0:2} {1:2} {2:8}'.format(m0, m1, fidelity)) 

# print('{0:2} {1:2} {2:8}'.format('m0', 'm1', 'fidelity'))
# for conf, psi_proj, psi_ref in zip(confs, psis_proj, psis_ref):
#     fidelity = np.round(np.abs(psi_proj.overlap(psi_ref))**2., 3)
#     m0, m1 = conf
#     print('{0:2} {1:2} {2:8}'.format(m0, m1, fidelity)) 
# # psi0 = rand_ket(2) #to be teleported
# # phi=bell_state('00')
# # qt.tensor(psi,phi)

# psi1=cnot(N=3, control=1, target=2)*psi0
# psi2 = snot(N=3, target=0)*psi1
import qutip as qt
from qutip_qip.circuit import QubitCircuit

print("Executing quantum simulation...")

# 1. Setting up the 3-qubit state correctly
psi = (qt.basis(2, 0) + qt.basis(2, 1)).unit() # This is the |+> state
state_0 = qt.basis(2, 0)
state = qt.tensor(psi, state_0, state_0) # Combined 3-qubit package

# 2. Building a 3-qubit circuit
teleportation = QubitCircuit(3, num_cbits=2)
teleportation.add_measurement("M0", targets=[0], classical_store=1)

# 3. Running the math
results = teleportation.run_statistics(state)

# 4. Printing clean output
print("\n--- RESULTS ---")
print("Probabilities of outcomes:", results.probabilities)
print("State vector dimensions:", results.final_states[0].dims)
print("State vector matrix shape:", results.final_states[0].shape)