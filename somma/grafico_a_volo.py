from matplotlib import pyplot as plt

snr = [0.378, 0.059, 0.004, 0.001, 1.554e-05, 0]

ax = plt.gca()
plt.plot([i for i in range(2, 8)], snr)
ax.set_title('SNR curve')
plt.xlabel("Number of digits")
plt.ylabel("SNR")
plt.show()