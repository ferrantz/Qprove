from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, IBMQ, execute, transpile
from numpy import pi

qreg_q = QuantumRegister(2, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.x(qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])

IBMQ.load_account()
num_shots_reale = 5000
BACKEND = 'ibmq_montreal' # 
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# provider_reale = IBMQ.get_provider(hub = 'ibm-q')
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators') # per simulatori
backend_reale = provider_reale.get_backend(BACKEND) 
# job_reale = execute(circuit, backend = backend_reale, shots=num_shots_reale)
c_basis = transpile(circuit, backend_reale, optimization_level=0)
job_reale = execute(c_basis, backend = backend_reale, shots=num_shots_reale)
# job_1 = backend_reale.retrieve_job('62bec0f1218cea59f65d7547')


diz_risposte_esatte = {'job_1': '10', 'job_2': '01'}