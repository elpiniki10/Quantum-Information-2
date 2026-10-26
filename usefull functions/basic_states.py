import qutip as qt

state0=qt.basis(2, 0)
state00=qt.tensor(qt.basis(2, 0),qt.basis(2, 0))
state11=qt.tensor(qt.basis(2,1),qt.basis(2,1))
state1=qt.basis(2,1)
state01=qt.tensor(qt.basis(2, 0),qt.basis(2, 1))
state10=qt.tensor(qt.basis(2, 1),qt.basis(2, 0))

state0_dm=qt.ket2dm(state0)
state1_dm=qt.ket2dm(state1)

state00_dm=qt.ket2dm(state00)
state01_dm=qt.ket2dm(state01)
state10_dm=qt.ket2dm(state10)
state11_dm=qt.ket2dm(state11)

bell00_dm=qt.ket2dm(qt.bell_state('00'))
bell10_dm=qt.ket2dm(qt.bell_state('10'))
bell01_dm=qt.ket2dm(qt.bell_state('01'))
bell11_dm=qt.ket2dm(qt.bell_state('11'))
