import random 
import cirq
from cirq import H, X, CNOT, measure

def make_oracle(q0, q1, secret_numbers):

    if secret_numbers[0]: # if the first number is 1 yield 
        yield[CNOT(q0,q1), X(q1)] # yield CNOT gate and X gate moments
        
    if secret_numbers[1]:
        yield CNOT(q0, q1) # if the second number is 1 yield CNOT gate
         
def make_deutsch_circuit(q0, q1, oracle):
    c = cirq.Circuit()
    c.append([X(q1), H(q1), H(q0)]) # append X gate and two H gates to the circuit
    c.append(oracle) # append oracle to circuit
    c.append([H(q0), measure(q0, key='result')]) # append H gate on first qubit and then a mesure function to determine the output.
    return c

def main():
    q0, q1 = cirq.LineQubit.range(2) #create 2 qubits   
    secret_numbers = [random.randint(0,1) for i in range(2)] #create list of two numbers
    oracle = make_oracle(q0, q1, secret_numbers) # create oracle moment to process the numbers in the list
    print('Secret function:\nf(x) = <{}>'.format(', '.join(str(e) for e in secret_numbers))) # print out list numbers
    circuit = make_deutsch_circuit(q0,q1, oracle) #create circuit 
    print("Circuit:") 
    print(circuit) 
    simulator = cirq.Simulator() # create simulator
    result = simulator.run(circuit)  #run circuit through simulator
    print('Result of f(0)⊕f(1):') 
    print(result) # print result

if __name__ == '__main__':
    main()