"""Recreates Fig. 4: random 2-qubit states are tested with the MBM criterion (33) and the
three strengthened CHSH inequalities (42, 43, 46).
Saves the numbers for every state in venn_results.txt (autosave every CHECKPOINT_EVERY states,
resumes after a crash). Μakes the Venn diagrams only from that file."""

from correlation import sum_PCC, CHSH1_value, CHSH2_value, CHSH3_value, is_entangled_PPT
from results_io import run_checkpointed
from venn_plot import make_venn_diagrams
import qutip as qt

NUM_STATES = 1000000
CHECKPOINT_EVERY = 100000# Save results every 
RESULTS_FILE = "venn_results.txt"
#observables
a = [qt.sigmax(), qt.sigmay()]
b = [qt.sigmax(), qt.sigmay()]


HEADER = ["state","ppt_ent","mbm","chsh1","chsh2","chsh3"]

def compute_row(i:int):
    """"Calculates and returns all quantities that will be saved for the i-th random state.
    The first column is the state index, the second is 1 if the state is entangled (PPT test), else 0, the third is |Cor{A1,B1}|+|Cor{A2,B2}| (entangled if >1), and the last three are the CHSH values."""
    Sab=qt.rand_dm([2, 2])
    return [i,int(is_entangled_PPT(Sab)),sum_PCC(Sab,a,b),
            CHSH1_value(Sab,a,b),CHSH2_value(Sab,a,b),CHSH3_value(Sab,a,b)]

if __name__ == "__main__":
    # results
    run_checkpointed(RESULTS_FILE,HEADER,NUM_STATES,compute_row,CHECKPOINT_EVERY)

    # graphs
    make_venn_diagrams(RESULTS_FILE)
