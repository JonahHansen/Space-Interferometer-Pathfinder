import numpy as np
import matplotlib.pyplot as plt

file = "HQvariance.npy"

a = np.load(file)
b = a.transpose()

array = np.zeros((180,180))

alt = 5e5
Om=0
b = 0.3e3
ra = np.radians(0)

for item in a:
    if item[0] == alt:
        if item[1] == b:
            if item[3] == Om:
                if item[4] == ra:
                    i = int(180/np.pi*item[2]+90)
                    dec = int(180/np.pi*item[5]+90)
                    max_acc = item[10]
                    array[i,dec] = max_acc

vmin = 1e-8
vmax = 1e-5

plt.imshow(array,origin="lower",cmap="viridis",extent=[-90,90,-90,90],vmin=vmin,vmax=vmax)
cbar = plt.colorbar()
cbar.set_label("Max acc", rotation=270, labelpad=15)
plt.xlabel("Declination (deg)")
plt.ylabel("Inclination (deg)")
plt.show()
