import qutip as qt
from sympy import plot
import numpy as np
import matplotlib.pyplot as plt
from correlation import sum_PCC

def depo_chan(rho:qt.Qobj,p):
        depo=rho*(1-p)+p*((qt.tensor(qt.qeye(2),qt.qeye(2)))/4)
        return depo

def bitflip_chan(rho:qt.Qobj,p):
        flip=(1-p)*rho+p*(qt.tensor(qt.sigmax(),qt.qeye(2)))*rho*(qt.tensor(qt.sigmax(),qt.qeye(2)))
        return flip
def figure_depolarising(state:qt.Qobj):
    p_vals=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    depresults=[]
    for p in p_vals:
            x=depo_chan(state,p)
            sumpcc=sum_PCC(x,Pauli_A,Pauli_B)
            depresults.append(sumpcc)
    return depresults

def figure_bitflip(state:qt.Qobj):
    p_vals=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    bitresults=[]
    for p in p_vals:
        x=bitflip_chan(state,p)
        sumpcc=sum_PCC(x,Pauli_A,Pauli_B)
        bitresults.append(sumpcc)
    return bitresults
p_vals=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
if __name__ == "__main__":
        
    bell00=qt.ket2dm(qt.bell_state('00'))
#state 2
    m = 0.06
    state_00 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    state_11 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))
    psi = np.sqrt(m) * state_00 + np.sqrt(1 - m) * state_11
    psi_m = qt.ket2dm(psi)

    Pauli_A = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    Pauli_B = [qt.sigmax(), qt.sigmay(), qt.sigmaz()]

    figbell=figure_depolarising(bell00)
    figm1=figure_depolarising(psi_m)
    figbell2=figure_bitflip(bell00)
    figm2=figure_bitflip(psi_m)

    #plot a
    plt.figure(figsize=(8,5))
    plt.plot(p_vals,figbell)
    plt.axhline(y=1, color='black', label='Entanglement Threshold')
    plt.plot(p_vals,figm1,linestyle='--')    
    plt.xlabel('p')
    plt.ylabel('Sum')
    plt.title('depolarising channel')
    plt.legend()
    plt.grid(True)
    plt.show()


   #plotb

    plt.figure(figsize=(8,5))
    plt.plot(p_vals,figbell2)
    plt.axhline(y=1, color='black', linestyle='--', label='Entanglement Threshold')
    plt.plot(p_vals,figm2,linestyle='--')  
    plt.xlabel('p')
    plt.ylabel('Sum')
    plt.title('Bit Flip Channel')
    plt.legend()
    plt.grid(True)
    plt.show()
