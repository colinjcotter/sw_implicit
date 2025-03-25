import numpy as np
import matplotlib.pyplot as plt
from comp_eady_data import *
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
import argparse
parser = argparse.ArgumentParser(description='Compressible Eady diagnostics plots')
parser.add_argument('--highres', action='store_true', help='plot control and high resolution results at once')

args = parser.parse_known_args()
args = args[0]

# low resolution result (ncolumns=30, nlayers=30)
rmsv0 = rmsv(n=30)
maxv0 = maxv(n=30)
kineticv0 = kineticv(n=30)
kineticuw0 = kineticuw(n=30)
potential0 = potential(n=30)
total0 = total(n=30)

# set time
time = []
imax = 30*24/2
i = 0
while i < imax:
    time.append(2.*i)
    i += 1

# find out when maxv0 reaches 3 m/s
i = 0
while i < imax:
    print("maxv0 = ", maxv0[i], "t = ", time[i])
    if maxv0[i] > 3:
        print("maxv0 reaches ", maxv0[i], "at t = ", time[i], ". Time is reset to 0.")
        breeding_time = time[i]
        breeding_step = i
        break
    i += 1

# reset time0 to 0 after breeding
i = 0
time0_mod = []
rmsv0_mod = []
kineticv0_mod = []
kineticuw0_mod = []
potential0_mod = []
total0_mod = []
while i < imax - breeding_step:
    time0_mod.append(time[i]/24)
    rmsv0_mod.append(rmsv0[i + breeding_step])
    total0_mod.append(total0[i + breeding_step])
    kineticuw0_mod.append(kineticuw0[i + breeding_step])
    kineticv0_mod.append(kineticv0[i + breeding_step])
    potential0_mod.append(potential0[i + breeding_step])
    i += 1

if args.highres:
    # high resolution result (ncolumns=60, nlayers=60)
    rmsv1 = rmsv(n=60)
    maxv1 = maxv(n=60)
    kineticv1 = kineticv(n=60)
    kineticuw1 = kineticuw(n=60)
    potential1 = potential(n=60)
    total1 = total(n=60)
    
    # find out when maxv1 reaches 3 m/s
    i = 0
    while i < imax:
        print("maxv1 = ", maxv1[i], "t = ", time[i])
        if maxv1[i] > 3:
            print("maxv1 reaches ", maxv1[i], "at t = ", time[i], ". Time is reset to 0.")
            breeding_time = time[i]
            breeding_step = i
            break
        i += 1

    # reset time1 to 0 after breeding
    i = 0
    time1_mod = []
    rmsv1_mod = []
    kineticv1_mod = []
    kineticuw1_mod = []
    potential1_mod = []
    total1_mod = []
    while i < imax - breeding_step:
        time1_mod.append(time[i]/24)
        rmsv1_mod.append(rmsv1[i + breeding_step])
        total1_mod.append(total1[i + breeding_step])
        kineticuw1_mod.append(kineticuw1[i + breeding_step])
        kineticv1_mod.append(kineticv1[i + breeding_step])
        potential1_mod.append(potential1[i + breeding_step])
        i += 1

    # high resolution result (ncolumns=90, nlayers=90)
    rmsv2 = rmsv(n=90)
    maxv2 = maxv(n=90)
    kineticv2 = kineticv(n=90)
    kineticuw2 = kineticuw(n=90)
    potential2 = potential(n=90)
    total2 = total(n=90)
    
    # find out when maxv2 reaches 3 m/s
    i = 0
    while i < imax:
        print("maxv2 = ", maxv2[i], "t = ", time[i])
        if maxv2[i] > 3:
            print("maxv2 reaches ", maxv2[i], "at t = ", time[i], ". Time is reset to 0.")
            breeding_time = time[i]
            breeding_step = i
            break
        i += 1

    # reset time2 to 0 after breeding
    i = 0
    time2_mod = []
    rmsv2_mod = []
    kineticv2_mod = []
    kineticuw2_mod = []
    potential2_mod = []
    total2_mod = []
    while i < imax - breeding_step:
        time2_mod.append(time[i]/24)
        rmsv2_mod.append(rmsv2[i + breeding_step])
        total2_mod.append(total2[i + breeding_step])
        kineticuw2_mod.append(kineticuw2[i + breeding_step])
        kineticv2_mod.append(kineticv2[i + breeding_step])
        potential2_mod.append(potential2[i + breeding_step])
        i += 1

    # high resolution result (ncolumns=120, nlayers=120)
    rmsv3 = rmsv(n=120)
    maxv3 = maxv(n=120)
    kineticv3 = kineticv(n=120)
    kineticuw3 = kineticuw(n=120)
    potential3 = potential(n=120)
    total3 = total(n=120)

    # find out when maxv3 reaches 3 m/s
    i = 0
    while i < imax:
        print("maxv3 = ", maxv3[i], "t = ", time[i])
        if maxv3[i] > 3:
            print("maxv3 reaches ", maxv3[i], "at t = ", time[i], ". Time is reset to 0.")
            breeding_time = time[i]
            breeding_step = i
            break
        i += 1

    # reset time3 to 0 after breeding
    i = 0
    time3_mod = []
    rmsv3_mod = []
    kineticv3_mod = []
    kineticuw3_mod = []
    potential3_mod = []
    total3_mod = []
    while i < imax - breeding_step:
        time3_mod.append(time[i]/24)
        rmsv3_mod.append(rmsv3[i + breeding_step])
        total3_mod.append(total3[i + breeding_step])
        kineticuw3_mod.append(kineticuw3[i + breeding_step])
        kineticv3_mod.append(kineticv3[i + breeding_step])
        potential3_mod.append(potential3[i + breeding_step])
        i += 1
        
# set parameters for visualisation
colors = ["black", "blue", "green", "red", "green", "darkorange", "brown", "yellowgreen", "purple"]
markers = ["+", "s", "o", "v", "^", "<", ">", "1", "2", "3"]

# plot rmsv
plt.figure(num=1)

if args.highres:
    plt.plot(time0_mod, rmsv0_mod, color=colors[0], linewidth=1., linestyle="solid", label = r'$N=30$')
    plt.plot(time1_mod, rmsv1_mod, color=colors[1], linewidth=1., linestyle="solid", label = r'$N=60$')
    plt.plot(time2_mod, rmsv2_mod, color=colors[2], linewidth=1., linestyle="solid", label = r'$N=90$')
    plt.plot(time3_mod, rmsv3_mod, color=colors[3], linewidth=1., linestyle="solid", label = r'$N=120$')
else:
    plt.plot(time0_mod, rmsv0_mod, color=colors[0], linewidth=1., linestyle="solid")

plt.grid(which="major", color="gray", linewidth=0.25, linestyle="solid")
# plt.title('rmsv in compressible fronts', fontsize=13)

if args.highres:
    plt.legend(prop={'size':10}, loc="lower right")

plt.xlabel(r't (days)', fontsize=14)
plt.xlim(0,25)
plt.ylim(0,50)
plt.ylabel(r'$v$ (m s$^{-1}$)', fontsize=14)
#plt.ylim(1.e-5,1.e-1)
#plt.yscale('log')

plt.savefig('rmsv.pdf')
#plt.show()

# plot peak
plt.figure(num=2)

if args.highres:
    plt.plot(time0_mod, rmsv0_mod, color=colors[0], linewidth=1., linestyle="solid", label = r'$N=30$')
    plt.plot(time1_mod, rmsv1_mod, color=colors[1], linewidth=1., linestyle="solid", label = r'$N=60$')
    plt.plot(time2_mod, rmsv2_mod, color=colors[2], linewidth=1., linestyle="solid", label = r'$N=90$')
    plt.plot(time3_mod, rmsv3_mod, color=colors[3], linewidth=1., linestyle="solid", label = r'$N=120$')
else:
    plt.plot(time0_mod, rmsv0_mod, color=colors[0], linewidth=1., linestyle="solid")

plt.grid(which="major", color="gray", linewidth=0.25, linestyle="solid")
# plt.title('rmsv in compressible fronts', fontsize=13)

if args.highres:
    plt.legend(prop={'size':10}, loc="upper left")
    
plt.xlabel(r't (days)', fontsize=14)
plt.xlim(5,10)
plt.ylim(20,50)
plt.ylabel(r'$v$ (m s$^{-1}$)', fontsize=14)
#plt.ylim(1.e-5,1.e-1)
#plt.yscale('log')

plt.savefig('rmsv_zoom.pdf')
#plt.show()


# plot energy
plt.figure(num=3)

if args.highres:
    l1, = plt.plot(time0_mod, total0_mod, color=colors[0], linewidth=2., linestyle="solid", label = r'$E \, (N=30)$')
    l2, = plt.plot(time0_mod, kineticuw0_mod, color=colors[0], linewidth=1., linestyle="dotted", label = r'$K_u \, (N=30)$')
    l3, = plt.plot(time0_mod, kineticv0_mod, color=colors[0], linewidth=1., linestyle="solid", label = r'$K_v \, (N=30)$')
    l4, = plt.plot(time0_mod, potential0_mod, color=colors[0], linewidth=1., linestyle="dashdot", label = r'$P \, (N=30)$')
    h1 = [l1, l2, l3, l4]
    # plt.plot(time1_mod, total1_mod, color=colors[1], linewidth=2., linestyle="solid")
    # plt.plot(time1_mod, kineticuw1_mod, color=colors[1], linewidth=1., linestyle="dotted")
    # plt.plot(time1_mod, kineticv1_mod, color=colors[1], linewidth=1., linestyle="solid")
    # plt.plot(time1_mod, potential1_mod, color=colors[1], linewidth=1., linestyle="dashdot")
    # plt.plot(time2_mod, total2_mod, color=colors[2], linewidth=2., linestyle="solid")
    # plt.plot(time2_mod, kineticuw2_mod, color=colors[2], linewidth=1., linestyle="dotted")
    # plt.plot(time2_mod, kineticv2_mod, color=colors[2], linewidth=1., linestyle="solid")
    # plt.plot(time2_mod, potential2_mod, color=colors[2], linewidth=1., linestyle="dashdot")
    l5, = plt.plot(time3_mod, total3_mod, color=colors[3], linewidth=2., linestyle="solid", label = r'$E \, (N=120)$')
    l6, = plt.plot(time3_mod, kineticuw3_mod, color=colors[3], linewidth=1., linestyle="dotted", label = r'$K_u \, (N=120)$')
    l7, = plt.plot(time3_mod, kineticv3_mod, color=colors[3], linewidth=1., linestyle="solid", label = r'$K_v \, (N=120)$')
    l8, = plt.plot(time3_mod, potential3_mod, color=colors[3], linewidth=1., linestyle="dashdot", label = r'$P \, (N=120)$')
    h2 = [l5, l6, l7, l8]
else:
    plt.plot(time0_mod, total0_mod, color=colors[0], linewidth=2., linestyle="solid", label = r'$E$')
    plt.plot(time0_mod, kineticuw0_mod, color=colors[0], linewidth=1., linestyle="dotted", label = r'$K_u$')
    plt.plot(time0_mod, kineticv0_mod, color=colors[0], linewidth=1., linestyle="solid", label = r'$K_v$')
    plt.plot(time0_mod, potential0_mod, color=colors[0], linewidth=1., linestyle="dashdot", label = r'$P$')
    
plt.grid(which="major", color="gray", linewidth=0.25, linestyle="solid")

# plt.title('energy in compressible fronts', fontsize=13)
if args.highres:
    lab1 = [h.get_label() for h in h1]
    lab2 = [h.get_label() for h in h2]
    leg1 = plt.legend(h1,lab1, loc="upper left")
    leg2 = plt.legend(h2,lab2, loc="lower left")
    plt.gca().add_artist(leg1)
else:
    plt.legend(prop={'size':10}, loc="upper left")

plt.xlabel(r't (days)', fontsize=14)
plt.xlim(0,25)
plt.ylim(-2.e14,2.e14)
plt.ylabel(r'Eenergy difference from the initial state', fontsize=14)

plt.savefig('energy.pdf')
#plt.show()

