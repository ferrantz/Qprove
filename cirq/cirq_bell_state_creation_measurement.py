import cirq

q_register = [cirq.LineQubit(i) for i in range(2)] # i due qubit dello Stato di Bell
# definisce il circuito con un H sul qubit 0 seguito da un CNOT
cirquit = cirq.Circuit([cirq.H(q_register[0]), cirq.CNOT(q_register[0], q_register[1])])
# misura i qubit
cirquit.append(cirq.measure(*q_register, key = 'z'))
print('Circuito')
print(cirquit)
sim = cirq.Simulator()
output = sim.run(cirquit, repetitions = 100)
print('Output della misurazione')
print(output.histogram(key = 'z')) # 0 sta per |00> mentre 3 per |11>