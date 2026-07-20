import numpy as np

import itertools
import qutip as qt
from qutip import basis, tensor, rand_ket, qeye
from qutip_qip.operations import snot, cnot, rx, rz
from qutip_qip.circuit import QubitCircuit,CircuitSimulator




teleportation = QubitCircuit(
    3, num_cbits=2, input_states=[r"\psi", "0", "0", "c0", "c1"]
)#3 qubits 2 classical bits for saving
#create bell 00
teleportation.add_gate("SNOT", targets=[1])
teleportation.add_gate("CNOT", targets=[2], controls=[1])
#alice operations
teleportation.add_gate("CNOT", targets=[1], controls=[0])
teleportation.add_gate("SNOT", targets=[0])
#alices measurments
teleportation.add_measurement("M0", targets=[0], classical_store=1)
teleportation.add_measurement("M1", targets=[1], classical_store=0)
#bobs corrections
teleportation.add_gate("X", targets=[2], classical_controls=[0])
teleportation.add_gate("Z", targets=[2], classical_controls=[1])
#bobs measurment???????????????????????????????
#teleportation.add_measurement("M2", targets=[2], classical_store=0)

teleportation.draw()

if __name__ == "__main__":# test
    alice = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    #a=qt.rand_ket(2)
    state = tensor(alice, basis(2, 0), basis(2, 0))
   
    state_final = teleportation.run(state)
    #print(state_final)
    final_results = teleportation.run_statistics(state)#computes every possible measurement outcome and their probabilities.
    #print("Possible states:   ",final_results.final_states,"probabilities:   " ,final_results.probabilities)
    bob = state_final.ptrace(2)#get only bobs qubit 
    print("fidelity",qt.fidelity(alice,bob))