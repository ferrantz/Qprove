from qiskit import *

IBMQ.load_account()
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
backend_reale = provider_reale.get_backend('ibmq_montreal')
job = backend_reale.retrieve_job('62bc1e5008f974c30c6643d6')

print('cambia')