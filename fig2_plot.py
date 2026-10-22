"""Fig. 2, step 2 (plots): made only from fig2_results.txt (run fig2_data.py first).
"""

import sys
import matplotlib.pyplot as plt
from results_io import load_results

RESULTS_FILE = "fig2_results.txt"

def plot_panel(data,channel:str,color,title,filename):
    """channel is 'bitflip' or 'depo' (column prefix in the results file)"""
    p_vals=data["p"]
    bell_vals=data[channel+"_bell_sum"]
    psi_vals=data[channel+"_psi_sum"]
    psi_ent=data[channel+"_psi_ent"].astype(bool)
    plt.figure(figsize=(8,5))
    plt.plot(p_vals,bell_vals,color=color,label=r'$|\phi_1\rangle\langle\phi_1|$ input')
    plt.plot(p_vals,psi_vals,color=color,linestyle='--',label=r'$|\psi\rangle\langle\psi|$ input')
    plt.axhline(y=1,color='black',linewidth=1,label='Entanglement threshold')
    # entangled (PPT) but not identified by the PCC criterion
    
    #plt.fill_between(p_vals,0,1,where=psi_ent&(psi_vals<=1),color='lightgray',alpha=0.6,
                     #label='Entangled but not identified')
    plt.xlim(0,1)
    plt.ylim(0,3)
    plt.xlabel('p')
    plt.ylabel(r'$\sum_{i=1}^{3}|\mathrm{Cor}\{A_i,B_i\}|$')
    plt.title(title)
    plt.legend()
    plt.savefig(filename,dpi=300,bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data=load_results(sys.argv[1] if len(sys.argv)>1 else RESULTS_FILE)
    #plot a
    plot_panel(data,'bitflip','tab:red','(a) Bit flip channel','fig2a_bitflip.png')
    #plot b
    plot_panel(data,'depo','tab:blue','(b) Depolarizing channel','fig2b_depolarizing.png')
