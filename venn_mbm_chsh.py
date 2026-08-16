from correlation import is_ent_MBM,is_ent_strenghenedCHSH1,is_ent_stenghenedCHSH3,is_ent_stenghenedCHSH2
from matplotlib_venn import venn2,venn3 
from matplotlib import pyplot as plt
import qutip as qt

def venn_ent():
    """Generates random two-qubit states and evaluates them for entanglement.
Returns four sets containing the indices of the states identified as 
entangled by the MBM criterion and three strengthened CHSH inequalities.
"""
    a = [qt.sigmax(), qt.sigmay()]
    b = [qt.sigmax(), qt.sigmay()]
    
    entangled_set_MBM = set()
    entangledCHSH1=set()
    entangledCHSH2=set()
    entangledCHSH3=set()
    
    num_states = 1000
    
    for i in range(num_states):

        Sab=qt.rand_dm([2, 2])
        
        if is_ent_MBM(Sab, a, b):
            entangled_set_MBM.add(i)
    # 42
        if is_ent_strenghenedCHSH1(Sab, a, b):
            entangledCHSH1.add(i)
                
    #43
        if is_ent_stenghenedCHSH2(Sab, a, b):
            entangledCHSH2.add(i)
                
        #46
        if is_ent_stenghenedCHSH3(Sab, a, b):
            entangledCHSH3.add(i)   
            
    return entangled_set_MBM,entangledCHSH1,entangledCHSH2,entangledCHSH3

mbm,c1,c2,c3=venn_ent()
chsh= c1 |c2|c3
total_identified = len(mbm | chsh)
total_chsh=len(chsh)

def format_percenta(x):
    if total_identified==0:
        return "0%"
    e=(x*100)/total_identified
    return f"{e:.2f}%"

def format_percentb(x):
    if total_chsh==0:
        return "0%"
    e=(x*100)/total_chsh
    return f"{e:.2f}%"

plt.figure(figsize=(8, 8))
venn = venn2(
    subsets=(mbm, chsh), 
    set_labels=('MBM', 'Any CHSH'),
    set_colors=('gold', 'silver'),
    subset_label_formatter=format_percenta
)

plt.title("a)Entangled states identified by any method")
plt.show()

plt.figure(figsize=(8, 8))
venn = venn3(
    subsets=(c1,c2,c3), 
    set_labels=('CHSH1', 'CHSH2','CHSH3'),
    set_colors=('gold', 'silver','red'),
    subset_label_formatter=format_percentb
)

plt.title("b)Entangled states identified by CHSH")
plt.show()
