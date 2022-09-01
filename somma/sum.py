# da: https://medium.com/@sashwat.anagolum/arithmetic-on-quantum-computers-addition-7e0d700f53ae

from qiskit import Aer, execute, QuantumCircuit, ClassicalRegister, QuantumRegister

first = input("Scrivi il primo numero binario con meno di 8 cifre: ")
second = input("Scrivi il secondo numero binario con meno di 8 cifre: ")
# prendiamo numeri con max 8 cifre perché il simulatore di qiskit riesce a gestire un massimo di 32 qubit, in particolare:
# 2n bit per storare i numeri,
# n c.d. 'carry bits',
# n + 1 bit classici per salvare il risultato dell'addizione
# non creiamo un nuovo register per salvare il risultato quantistico perché sovrascriviamo il secondo numero con la somma
# dei due numeri
l = len(first)
l_2 = len(second)
if l > l_2:
     n = l
else:
     n = l_2
# si fa il controllo per vedere quale numero è maggiore e si usa la lunghezza del più lungo per settare le dimensioni dei
# registri, dopodiché si combinano tutti i register e si crea il circuito quantistico

a = QuantumRegister(n) # per il primo numero 
b = QuantumRegister(n+1) # per il secondo numero 
c = QuantumRegister(n) # per i carry bits
cl = ClassicalRegister(n+1) # il register classico ha n + 1 bit, ed è usato per fare la somma leggibile
#Combining all of them into one quantum circuit
qc = QuantumCircuit(a, b, c, cl)

# setting up dei registri coi valori presi in input
for i in range(l):
    if first[i] == "1":
       qc.x(a[l - (i+1)]) #Flip the qubit from 0 to 1
for i in range(l_2):
   if second[i] == "1":
      qc.x(b[l_2 - (i+1)]) #Flip the qubit from 0 to 1

for i in range(n-1):
    qc.ccx(a[i], b[i], c[i+1])
    qc.cx(a[i], b[i])
    qc.ccx(c[i], b[i], c[i+1])
    
# per l'ultima iterazione col carry gate, invece di salvare il risultato in c[n], usiamo b[n], motivo per cui c ha solo
# n bits, con c[n-1] che è l'ultimo carry bit
# NON SO SE VA NEL FOR
    qc.ccx(a[n-1], b[n-1], b[n])
    qc.cx(a[n-1], b[n-1])
    qc.ccx(c[n-1], b[n-1], b[n])

# a questo punto si fa il reverse di tutte le operazioni finora fatte per essere certi che i sum gates che a breve saranno
# implementati ricevano in pasto gli input corretti, e per poi fare il reverse del carry register al suo stato originale

# reverse delle operazioni performate su b[n-1]
qc.cx(c[n-1], b[n-1])

for i in range(n-1):
    qc.ccx(c[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
    qc.cx(a[(n-2)-i], b[(n-2)-i])
    qc.ccx(a[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
    # queste due operazioni fanno da sum gate; se un control bit è allo stato |1>, allora il target bit b[(n-2)-i] è flippato
    qc.cx(c[(n-2)-i], b[(n-2)-i])
    qc.cx(a[(n-2)-i], b[(n-2)-i])

# dopo aver performato l'inverse carry e l'operazione di reverse, il risultato dell'addizione deve essere storato nel 
# registro b. Comunque, per vedere il risultato, dobbiamo misurare i qubit e copiare il risultato nel registro classico cl

for i in range(n+1):
    qc.measure(b[i], cl[i])

num_shots = 2000 
selected_backend = "qasm_simulator"
job = execute(qc, Aer.get_backend(selected_backend), shots=num_shots)
job_stats = job.result().get_counts()
print(job_stats)