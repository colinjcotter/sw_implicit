import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# raw data
peak0 = 41.99046951137377
peak1 = 42.946515706506354
peak2 = 43.510722425503104
peak3 = 43.61394025327927

peaknorm0 = peak3-peak0
peaknorm1 = peak3-peak1
peaknorm2 = peak3-peak2

peaknorm = [peaknorm0, peaknorm1, peaknorm2]

# set time
L = 1000000.
nx = [30, 60, 90]
dx = []
for i in nx:
    m = 2*L/i
    dx.append(m)

print(dx)

# set parameters for visualisation
colors = ["black", "red", "blue", "green", "magenta", "darkorange", "brown", "yellowgreen", "purple"]
markers = ["+", "s", "o", "v", "^", "<", ">", "1", "2", "3"]

# plot rhonorm
plt.figure(num=1, figsize=(5,5))
plt.plot(dx, peaknorm, color=colors[0], marker=markers[0], linewidth=1., linestyle="solid")
plt.grid(which="minor", color="gray", linewidth=0.25, linestyle="solid")
plt.xscale("log")
plt.yscale("log")
plt.title('', fontsize=13)

#plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r'$\Delta x$', fontsize=14)
plt.xlim(2.0e4, 7.0e4)
#plt.ylim(0,50)
plt.ylabel(r'$v$ (m s$^{-1}$)', fontsize=14)
#plt.ylim(1.e-5,1.e-1)
#plt.yscale('log')

plt.savefig('peaknorm.pdf')
plt.show()

