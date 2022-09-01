from qiskit import QuantumCircuit, Aer, execute, IBMQ, QuantumRegister, ClassicalRegister
# from matplotlib import pyplot as plt

qr = QuantumRegister(2) # create register to store bits
cr = ClassicalRegister(2)
qc = QuantumCircuit(qr, cr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.x(qr[0])
qc.barrier()
qc.measure(qr, cr)

num_shots = 5000 
selected_backend = "statevector_simulator"
job = execute(qc, Aer.get_backend(selected_backend), shots=num_shots)
job_stats = job.result().get_counts()
print("statevector_simulator: " + str(job_stats))
qc.draw(output='mpl', filename=r'C:\\Users\\italo\\Desktop\\Qprove\\entanglement\\figure\\figura_esperimento_3.png')
#-------------------------------------------------------------------------------------------
# transpiler

# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from numpy import pi

# qreg_q = QuantumRegister(2, 'q')
# creg_c0 = ClassicalRegister(2, 'c0')
# circuit = QuantumCircuit(qreg_q, creg_c0)

# circuit.rz(1.5707963267948966, qreg_q[0])
# circuit.sx(qreg_q[0])
# circuit.rz(1.5707963267948966, qreg_q[0])
# circuit.cx(qreg_q[0], qreg_q[1])
# circuit.measure(qreg_q[0], creg_c0[0])
# circuit.measure(qreg_q[1], creg_c0[1])

# num_shots = 5000 
# selected_backend = "statevector_simulator"
# job = execute(circuit, Aer.get_backend(selected_backend), shots=num_shots)
# job_stats_transpiler = job.result().get_counts()
# print('risultati transpiler in locale: ' + str(job_stats_transpiler))
# circuit.draw(output='mpl', filename=r'C:\\Users\\italo\\Desktop\\Qprove\\entanglement\\figure\\bell_state_transpiled.png')


# # in remoto

IBMQ.load_account()
BACKEND = 'ibmq_montreal' # 
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# provider_reale = IBMQ.get_provider(hub = 'ibm-q')
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators') # per simulatori
backend_reale = provider_reale.get_backend(BACKEND) 
job_reale = execute(qc, backend = backend_reale, shots=num_shots)