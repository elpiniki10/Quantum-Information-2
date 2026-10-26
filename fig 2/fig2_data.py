"""Fig. 2, step 1 (data). It takes two input states:
the Bell state |φ₁⟩
the non-maximally entangled state |ψ⟩ = √μ|00⟩ + √(1−μ)|11⟩, with μ = 0.06
Each state is sent through a quantum channel acting on qubit A, for error probability p from 0 to 1.
Two channels are used: depolarizing (depo_chan) and bit flip (bitflip_chan).
After the channel, it computes Σᵢ |Cor{Λᵢ, Λᵢ}| over the three Pauli matrices using sum_PCC from correlation.py,
and whether the output is entangled (PPT test). Results go to fig2_results.txt.
Make the figures afterwards with fig2_plot.py.
"""

import qutip as qt
import numpy as np
from correlation import sum_PCC, is_entangled_PPT
from results_io import run_checkpointed
Pauli_A = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
Pauli_B = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
p_vals = np.linspace(0.0, 1.0, 1001)#continual
RESULTS_FILE = "fig2_results.txt"
CHECKPOINT_EVERY = 1000

m = 0.06
bell00=qt.ket2dm(qt.bell_state('00'))
#state 2
state_00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
state_11 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
psi = np.sqrt(m) * state_00 + np.sqrt(1 - m) * state_11
psi_m = qt.ket2dm(psi)

def depo_chan(rho:qt.Qobj,p):
        """Depolarizing channel on qubit A (eq 27): (1-p)ρ + p·(I/2 ⊗ ρB)"""
        depo = (1-p)*rho + p*qt.tensor(qt.qeye(2)/2, rho.ptrace(1))
        return depo

def bitflip_chan(rho:qt.Qobj,p):
        """Bit flip channel on qubit A (eq 28)"""
        flip=(1-p)*rho+p*(qt.tensor(qt.sigmax(),qt.qeye(2)))*rho*(qt.tensor(qt.sigmax(),qt.qeye(2)))
        return flip

# columns of the results file: Σ|Cor| and PPT entanglement (1/0) for every channel/input pair
HEADER = ["p",
          "bitflip_bell_sum","bitflip_bell_ent","bitflip_psi_sum","bitflip_psi_ent",
          "depo_bell_sum","depo_bell_ent","depo_psi_sum","depo_psi_ent"]

def compute_row(i:int):
    """performing calculations across the horizontal axis (rows) of a data table"""
    p=p_vals[i]
    row=[p]
    for channel in (bitflip_chan,depo_chan):
        for state in (bell00,psi_m):
            x=channel(state,p)
            row+=[sum_PCC(x,Pauli_A,Pauli_B),int(is_entangled_PPT(x))]
    return row

if __name__ == "__main__":
    # results -> fig2_results.txt (autosaved, resumes after a crash;
    # delete the file to recompute, e.g. after changing m or p_vals)
    run_checkpointed(RESULTS_FILE,HEADER,len(p_vals),compute_row,CHECKPOINT_EVERY)
