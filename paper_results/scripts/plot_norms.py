import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# raw data
rhonorm_day2 = [4.851023026764386e-11, 1.1195741227660507e-11, 4.482512317593378e-12] 
thetanorm_day2 = [3.2438310858160374e-09, 7.673869550181599e-10, 3.160584608724246e-10] 
unorm_day2 = [2.8317768380407765e-08, 7.345161295319053e-09, 3.7218518198937856e-09] 
rhonorm_day4 = [6.820591092184873e-11, 1.73135917105601e-11, 6.853432008255343e-12] 
thetanorm_day4 = [1.638263581453138e-08, 3.773145057367131e-09, 1.5736164784968871e-09] 
unorm_day4 = [1.4067368563273618e-07, 3.7911993555131964e-08, 2.0369392201188405e-08] 
rhonorm_day7 = [6.022528913331979e-09, 3.106242547075357e-09, 2.042232447098761e-09]
thetanorm_day7 = [2.0798667192024733e-06, 1.2766181161068075e-06, 8.57961185173301e-07]
unorm_day7 = [1.332079190543768e-05, 9.771795549437306e-06, 6.647067853381047e-06]
rhonorm_day11 = [1.0513849335768963e-08, 7.150645508345121e-09, 5.916494006057886e-09] 
thetanorm_day11 = [4.400137899353648e-06, 2.760945627563053e-06, 2.1545690726393575e-06] 
unorm_day11 = [1.786936023863374e-05, 9.756765227934036e-06, 7.43387374869159e-06] 

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

# # plot rhonorm
# plt.figure(num=1, figsize=(6,9))
# plt.plot(dx, rhonorm_day2, color=colors[0], marker=markers[0], linewidth=1., linestyle="solid", label="DAY 2")
# plt.plot(dx, rhonorm_day4, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashed", label="DAY 4")
# plt.plot(dx, rhonorm_day7, color=colors[0], marker=markers[0], linewidth=1., linestyle="dotted", label="DAY 7")
# plt.plot(dx, rhonorm_day11, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashdot", label="DAY 11")
# plt.grid(which="minor", color="gray", linewidth=0.25, linestyle="solid")
# plt.xscale("log")
# plt.yscale("log")
# # plt.title('rhonorm', fontsize=13)

# plt.legend(prop={'size':10}, loc="lower right")

# plt.xlabel(r'$\Delta$x', fontsize=14)
# #plt.xlim(0,25)
# #plt.ylim(0,50)
# plt.ylabel(r'rhonorm', fontsize=14)
# #plt.ylim(1.e-5,1.e-1)
# #plt.yscale('log')

# plt.savefig('rhonorm.pdf')
# plt.show()

# plot thetanorm
plt.figure(num=2, figsize=(5,9))
plt.plot(dx, thetanorm_day2, color=colors[0], marker=markers[0], linewidth=1., linestyle="solid", label="DAY 2")
plt.plot(dx, thetanorm_day4, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashed", label="DAY 4")
plt.plot(dx, thetanorm_day7, color=colors[0], marker=markers[0], linewidth=1., linestyle="dotted", label="DAY 7")
plt.plot(dx, thetanorm_day11, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashdot", label="DAY 11")
plt.grid(which="minor", color="gray", linewidth=0.25, linestyle="solid")
plt.xscale("log")
plt.yscale("log")
# plt.title('thetanorm', fontsize=13)

plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r'$\Delta$x', fontsize=14)
#plt.xlim(0,25)
#plt.ylim(0,50)
plt.ylabel(r'$L_2$ norms in $\theta$', fontsize=14)
#plt.ylim(1.e-5,1.e-1)
#plt.yscale('log')

plt.savefig('thetanorm.pdf')
plt.show()

# plot unorm
plt.figure(num=3, figsize=(5,9))
plt.plot(dx, unorm_day2, color=colors[0], marker=markers[0], linewidth=1., linestyle="solid", label="DAY 2")
plt.plot(dx, unorm_day4, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashed", label="DAY 4")
plt.plot(dx, unorm_day7, color=colors[0], marker=markers[0], linewidth=1., linestyle="dotted", label="DAY 7")
plt.plot(dx, unorm_day11, color=colors[0], marker=markers[0], linewidth=1., linestyle="dashdot", label="DAY 11")
plt.grid(which="minor", color="gray", linewidth=0.25, linestyle="solid")
plt.xscale("log")
plt.yscale("log")
# plt.title('unorm', fontsize=13)

plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r'$\Delta$x', fontsize=14)
#plt.xlim(0,25)
#plt.ylim(0,50)
plt.ylabel(r'$L_2$ norms in $\mathbf{u}$', fontsize=14)
#plt.ylim(1.e-5,1.e-1)
#plt.yscale('log')

plt.savefig('unorm.pdf')
plt.show()
