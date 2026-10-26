"""Makes the Venn diagrams of Fig. 4 using only the saved results (venn_results.txt).
No states are generated here, so it runs in seconds and can be re-run as often as you like.
Region labels are exclusive regions, so they add up to 100%.
"""

from math import sqrt
import sys
from matplotlib import pyplot as plt
from matplotlib_venn import venn2,venn3_unweighted
from results_io import load_results

RESULTS_FILE = "venn_results.txt"

def venn_ent(data):
    """Sets with the indices of the entangled states identified by the MBM criterion
    and by the three strengthened CHSH inequalities"""
    ent=data[data["ppt_ent"]==1]   # keeps only the rows where ppt says its entangled(1 for ent 0 not)
    idx=ent["state"].astype(int)# Get their state indices as integers
    entangled_set_MBM = set(idx[ent["mbm"]>1])# Create a set of states detected by MBM
    entangledCHSH1 = set(idx[ent["chsh1"]>sqrt(2)])   # 42
    entangledCHSH2 = set(idx[ent["chsh2"]>sqrt(2)])   # 43
    entangledCHSH3 = set(idx[ent["chsh3"]>sqrt(2)])   # 46
    return len(ent),entangled_set_MBM,entangledCHSH1,entangledCHSH2,entangledCHSH3

def percent(total):
    """Region label: % of total, empty regions are left blank"""
    return lambda x: "" if total==0 or x==0 else f"{x*100/total:.2f}%"

def print_summary(n_states,n_ent,mbm,c1,c2,c3):
    chsh=c1|c2|c3
    identified=mbm|chsh
    print(f"states in file:              {n_states}")
    print(f"entangled (PPT):             {n_ent}")
    print(f"identified by any method:    {len(identified)} ({100*len(identified)/max(n_ent,1):.2f}% of entangled)")
    print(f"  MBM (33):                  {len(mbm)}")
    print(f"  any CHSH:                  {len(chsh)}")
    print(f"    CHSH (42):               {len(c1)}")
    print(f"    Cov CHSH (43):           {len(c2)}")
    print(f"    PCC CHSH (46):           {len(c3)}")
    print(f"  (43) subset of (46):       {c2<=c3}")

def make_venn_diagrams(path=RESULTS_FILE):
    data=load_results(path)
    n_ent,mbm,c1,c2,c3=venn_ent(data)
    chsh= c1 |c2|c3
    total_identified = len(mbm | chsh)
    total_chsh=len(chsh)
    print_summary(len(data),n_ent,mbm,c1,c2,c3)

    #a) MBM vs any strengthened CHSH, % of all identified states
    plt.figure(figsize=(8, 8))
    venn2(
        subsets=(mbm, chsh),
        set_labels=('MBM (33)', 'Any CHSH'),
        set_colors=('gold', 'silver'),
        subset_label_formatter=percent(total_identified)
    )
    plt.title("a) Entangled states identified by any method")
    plt.savefig("a_venn_diagram.png", dpi=300, bbox_inches='tight')
    plt.show()

    #b) the "any CHSH" set split into the three inequalities, % of states identified by any CHSH
    # equal circles: the sets are nearly nested, so area-proportional circles hide the labels
    plt.figure(figsize=(8, 8))
    venn3_unweighted(
        subsets=(c1,c2,c3),
        set_labels=('CHSH (42)', 'Cov CHSH (43)','PCC CHSH (46)'),
        set_colors=('red', 'green','blue'),
        subset_label_formatter=percent(total_chsh)
    )
    plt.title("b) Entangled states identified by the strengthened CHSH inequalities")
    plt.savefig("b_venn_diagram.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    make_venn_diagrams(sys.argv[1] if len(sys.argv)>1 else RESULTS_FILE)
