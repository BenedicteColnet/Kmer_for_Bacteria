import numpy as np
import matplotlib.pyplot as plt

with_dic_initialization = np.load('with_dic_initialization_k_13.npy')
without_dic_initialization = np.load('without_dic_initialization_k_13.npy')

plt.plot(with_dic_initialization, color = 'r', label='With dictionary initialization')
plt.plot(without_dic_initialization, color = 'b', label='Without dictionary initialization')
plt.ylabel('time(s)')
plt.xlabel('k')
plt.legend(loc='best')
plt.grid()
plt.show()