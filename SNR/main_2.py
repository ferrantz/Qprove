import pandas as pd
from matplotlib import pyplot as plt

d = {'number of gates': [2, 3, 4, 5], 'a': [0.1974, 0.061, 0.0224, 0.004], 
        'b': [0.1030182, 0.06244108, 0.03894719999999999, 0.026227759999999996], 
            'c': [8, 16, 32, 64], 'potenza_segnale': [0.024675, 0.0038125, 0.0007, 6.25e-05], 
                'potenza_rumore': [0.012877275, 0.0039025675, 0.0012170999999999996, 0.00040980874999999993], 
                    'SNR': [1.9161662696494404, 0.9769209629301735, 0.5751376222167449, 0.1525101648024841]}

data = pd.DataFrame(d)

ax = plt.gca()

data.plot(kind='line', x='number of gates', y='SNR', ax=ax)
ax.set_title('SNR curve')
ax.set_xticks(d['number of gates'])
plt.show()