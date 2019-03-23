import pandas as pd
import astropy.constants as const
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_json("bigfile.json","records")

r_orba = 500e3+const.R_earth.value
r_orbb = 1000e3+const.R_earth.value

ra = 0
Om = 0
R_orb = r_orba
delta_r_max = 0.3e3

data2 = data[(data["ra"] == ra) & (data["Om_0"] == Om) & (data["R_orb"] == R_orb) & (data["Delta_r"] == delta_r_max)]

data3 = data2.sort_values(by=['inc_0', 'dec']).reset_index()



def array_plot(z):
    inc_ls = np.radians(np.linspace(0,90,360))
    dec_ls = np.radians(np.linspace(-90,90,360))
    rnd = 5

    array = np.zeros((360,360))
    for i in range(360):
        for j in range(360):
            print(i,j)
            inc = inc_ls[i]
            dec = dec_ls[j]
            item = data2[(data2["inc_0"].round(rnd)==round(inc,rnd)) & (data2["dec"].round(rnd)==round(dec,rnd))]
            array[i,j] = item[z]
    return array

z = "Max_a_total"

#plt.imshow(array_plot(z))
array = np.zeros((360,360))
for i in range(360):
    for j in range(360):
        print(i,j)
        item = data2.iloc[i*360 + j]
        array[i,j] = np.log10(item[z])

plt.imshow(array,origin="lower",cmap="viridis",extent=[-90,90,0,90])
cbar = plt.colorbar()
cbar.set_label('Maximum accceleration m/s/s', rotation=270, labelpad=15)
plt.xlabel("Declination (deg)")
plt.ylabel("Inclination (deg)")

plt.show()
