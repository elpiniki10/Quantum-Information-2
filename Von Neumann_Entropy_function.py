"""
5. von Neumann entropy of a density matrix
"""
#make bool
import qutip as qt
from qutip import Qobj

def vonneumann_entropy(rho:Qobj)-> float:
    entropy = qt.entropy_vn(rho, base=2)
    return entropy
if __name__ == "__main__": #oxi
    #tests
    one = qt.basis(2, 1)
    rho1 = qt.ket2dm(one)
    print(vonneumann_entropy(rho1))
    zero = qt.basis(2, 0)
    rho2 = qt.ket2dm(zero)
    print(vonneumann_entropy(rho2))
    rho_mixed = 0.5 * qt.ket2dm(zero) + 0.5 * qt.ket2dm(one)
    print(vonneumann_entropy(rho_mixed))
    print(vonneumann_entropy(1))