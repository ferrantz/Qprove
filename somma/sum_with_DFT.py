import math
from qiskit import Aer, execute, QuantumCircuit, ClassicalRegister, QuantumRegister

first = input("Inserisci il primo numero con meno di 10 cifre: ")
second = input("Inserisci il primo numero con meno di 10 cifre: ")

# le cifre devono essere massimo 9 perché:
# n + 1 qubit sono per il primo numero;
# n + 1 qubit sono per il secondo numero;
# n+ 1 qubit servono per salvare l'output finale

l_1 = len(first)
l_2 = len(second)
# si fa caso se il primo numero è minore del secondo e nel caso si fa padding di 0
if l_2 > l_1:
    first,second = second, first
    l_2, l_1 = l_1, l_2
second = ("0")*(l_1-l_2) + second

# creiamo la prima funzione che converte il primo numero in uno stato appropriato per l'esercizio di addizione convertendolo
# in una trasformata di Fourier quantistica. 

def createInputState(qc, reg, n, pie):
    """
    Apply one Hadamard gate to the nth qubit of the quantum register               
    reg, and then apply repeated phase rotations with parameters  
    being pi divided by increasing powers of two.
    (Per rotazione della fase ci si intende al Controlled-U Gate)
    """
    qc.h(reg[n])    
    for i in range(0, n):
        qc.cu1(pie/float(2**(i+1)), reg[n-(i+1)], reg[n])

# a questo punto scriviamo un'altra funzione che performa le rotazioni controllate sui qubit.

def evolveQFTState(qc, reg_a, reg_b, n, pie):
    """
    Evolves the state |F(ψ(reg_a))> to |F(ψ(reg_a+reg_b))> using the     
    quantum Fourier transform conditioned on the qubits of reg_b.
    Apply repeated phase rotations with parameters being pi divided 
    by increasing powers of two.
    """
    for i in range(0, n+1):
        qc.cu1(pie/float(2**(i)), reg_b[n-i], reg_a[n])

# a questo punto occorre convertire la somma a + b, salvata nel registro a, dalla sua trasformata di Fourier quantistica
# ψ(a + b), alla forma a + b. Per farlo costruiamo una funzione che produce l'inverso di una QFT che svolge le stesse
# operazioni di createInputState() ma in ordine opposto. 

def inverseQFT(qc, reg, n, pie):
    """
    Svolge l'inverso di un QFT su un registro reg. Applica ripetute rotazioni della fase con parametri pi diviso per
    potenze decrescenti di due, e poi applica un gate di Hadamard all'ennesimo qubit del registro reg
    """
    for i in range(0, n):
        qc.cu1(-1*pie/float(2**(n-i)), reg[i], reg[n])
    qc.h(reg[n])

# creiamo i registri di cui abbiamo bisogno e salviamo i nostri numeri lì dentro.

def add(first, second, n):
    pie = math.pi
    a = QuantumRegister(n+1, "a") # conserva il primo numero 
    b = QuantumRegister(n+1, "b") # conserva il secondo numero 
    cl = ClassicalRegister(n+1, "cl") # conserva il risultato finale 
    qc = QuantumCircuit(a, b, cl, name="qc") # tutti i registri sono 'messi' qui
    # Flip the corresponding qubit in register a if a bit in the string first is a 1
    for i in range(0, n):
        if first[i] == "1":
            qc.x(a[n-(i+1)])
    #Flip the corresponding qubit in register b if a bit in the 
    #string second is a 1
    for i in range(0, n):
        if second[i] == "1":
            qc.x(b[n-(i+1)])
    # calcola la trasformata di Fourier su un registro a
    for i in range(0, n+1):
        createInputState(qc, a, n-i, pie)
    # aggiunge i due numeri facendo evolvere la trasformata di Fourier F(ψ(reg_a)) > a |F(ψ(reg_a+reg_b))>
    for i in range(0, n+1):
        evolveQFTState(qc, a, b, n-i, pie)
    # calcola l'inverso della trasformata di Fourier per il registro a
    for i in range(0, n+1):
        inverseQFT(qc, a, i, pie)
    # misuriamo i qubit nel registro a e salviamo i risultati nel registro classico cl
    for i in range(0, n+1):
        qc.measure(a[i], cl[i])

    num_shots = 2000 
    selected_backend = "qasm_simulator"
    job = execute(qc, Aer.get_backend(selected_backend), shots=num_shots)
    job_stats = job.result().get_counts()
    print(job_stats)

if __name__ == '__main__':
    n = len(second)
    add(first, second, n)