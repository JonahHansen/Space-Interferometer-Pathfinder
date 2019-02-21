import numpy as np
import matplotlib.pyplot as plt


def plotacc(index,array,title,clabel):
    len_ix = np.shape(a)[0]
    len_iy = np.shape(a)[1]
    result = np.zeros((len_ix,len_iy))
    plt.figure(index)
    for ix in range(len_ix):
       for iy in range(len_iy):
          item = a[ix,iy]
          result[int(item[0]),int(item[1])] = np.log10(item[(index % 4)+2])
    print(np.shape(result))
    plt.imshow(result,origin="lower",cmap="viridis",extent=[-90,90,0,90],vmin=-9)
    cbar = plt.colorbar()
    cbar.set_label(clabel, rotation=270, labelpad=15)
    plt.xlabel("Declination (deg)")
    plt.ylabel("Inclination (deg)")
    #yticks = np.linspace(0,90,len_ix)
    #xticks = np.linspace(-90,90,len_iy)
    #plt.xticks(xticks)
    #plt.yticks(yticks)
    plt.title(title)



a = np.load("s_acc_variance_alt1000.npy")
b = np.load("s_acc_variance_alt500.npy")

plotacc(0,a,"Altitude 1000km, Deputy 1",'Log maximum accceleration in s direction')
plotacc(1,a,"Altitude 1000km, Deputy 2",'Log maximum accceleration in s direction')
plotacc(2,a,"Altitude 1000km, Deputy 1",'Log delta_v in s direction')
plotacc(3,a,"Altitude 1000km, Deputy 2",'Log delta_v in s direction')
plotacc(4,b,"Altitude 500km, Deputy 1",'Log maximum accceleration in s direction')
plotacc(5,b,"Altitude 500km, Deputy 2",'Log maximum accceleration in s direction')
plotacc(6,b,"Altitude 500km, Deputy 1",'Log delta_v in s direction')
plotacc(7,b,"Altitude 500km, Deputy 2",'Log delta_v in s direction')

plt.show()
