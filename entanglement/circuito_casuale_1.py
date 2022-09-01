from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, Aer

q = QuantumRegister(5,'q')
c = ClassicalRegister(5,'c')


circuit = QuantumCircuit(q,c)

circuit.h(q[0]) # Hadamard gate 
circuit.cx(q[0],q[1]) # CNOT gate
circuit.h(q[1]) # Hadamard gate 
circuit.cx(q[1],q[2]) # CNOT gate
circuit.h(q[2]) # Hadamard gate 
circuit.cx(q[2],q[3]) # CNOT gate
circuit.h(q[3]) # Hadamard gate 
circuit.cx(q[3],q[4]) # CNOT gate
circuit.barrier()
circuit.measure(q,c) # Qubit Measurment
circuit.draw(output='mpl', filename = r'C:\\Users\\italo\\Desktop\\Qprove\\entanglement\\figure_circuito_casuale\\circuito_casuale_2.png')
# print(circuit)
# print(circuit.depth())
num_shots = 5000 
selected_backend = "statevector_simulator"
job = execute(circuit, Aer.get_backend(selected_backend), shots=num_shots)
job_stats = job.result().get_counts()
print(job_stats)


# IBMQ.load_account()
# BACKEND = 'ibmq_montreal' 
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# # provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
# # provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators') # per simulatori
# backend_reale = provider_reale.get_backend(BACKEND) 
# job_reale = execute(circuit, backend = backend_reale, shots=num_shots)

# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from numpy import pi

# qreg_q = QuantumRegister(27, 'q')
# creg_c = ClassicalRegister(3, 'c')
# circuit = QuantumCircuit(qreg_q, creg_c)

# circuit.rz(1.5707963267948966, qreg_q[0])
# circuit.sx(qreg_q[0])
# circuit.rz(1.5707963267948966, qreg_q[0])
# circuit.cx(qreg_q[0], qreg_q[1])
# circuit.rz(1.5707963267948966, qreg_q[1])
# circuit.sx(qreg_q[1])
# circuit.rz(1.5707963267948966, qreg_q[1])
# circuit.cx(qreg_q[1], qreg_q[2])
# circuit.barrier(qreg_q[0], qreg_q[1], qreg_q[2])
# circuit.measure(qreg_q[0], creg_c[0])
# circuit.measure(qreg_q[1], creg_c[1])
# circuit.measure(qreg_q[2], creg_c[2])
print(circuit)