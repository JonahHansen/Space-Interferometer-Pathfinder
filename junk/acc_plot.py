import numpy as np
import matplotlib.pyplot as plt


def plotacc(index,array,title,clabel):
    len_ix = np.shape(a)[0]
    len_iy = np.shape(a)[1]
    result = np.zeros((len_ix,len_iy))
    plt.figure(index)
    for ix in range(len_ix):
       for iy in range(len_iy):
          item = array[ix,iy]
          result[int(item[0]),int(item[1])] = item[(index % 6)+2]
    print(np.shape(result))
    if ((index % 6) < 3):
        vmin = 1e-9
        vmax = 5e-6
    else:
        vmin = 1e-7
        vmax = 1e-2

    plt.imshow(result,origin="lower",cmap="viridis",extent=[-90,90,0,90],vmin=vmin,vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label(clabel, rotation=270, labelpad=15)
    plt.xlabel("Declination (deg)")
    plt.ylabel("Inclination (deg)")
    #yticks = np.linspace(0,90,len_ix)
    #xticks = np.linspace(-90,90,len_iy)
    #plt.xticks(xticks)
    #plt.yticks(yticks)
    plt.title(title)

def plot_ave_del_v(idx,array,title):
    len_ix = np.shape(a)[0]
    len_iy = np.shape(a)[1]
    result = np.zeros((len_ix))
    plt.figure(idx)
    for ix in range(len_ix):
        item = array[ix]
        sum_del_v = 0
        weights = 0
        for iy in range(len_iy):
            weight = np.cos(item[iy,1])
            sum_del_v += item[iy,((idx % 10)+2)]*weight
            weights += weight
        result[int(item[0,0])] = sum_del_v/weights
    plt.bar(np.linspace(0,90,len_ix),result)
    plt.xlabel("Inclination (deg)")
    plt.ylabel("Delta_V (m/s)")
    plt.title(title)

def plot_max_ac(idx,array,title):
    len_ix = np.shape(a)[0]
    len_iy = np.shape(a)[1]
    result = np.zeros((len_ix))
    plt.figure(idx)
    for ix in range(len_ix):
        item = array[ix]
        result[int(item[0,0])] = np.max(item[:,((idx % 10)+2)])
    plt.bar(np.linspace(0,90,len_ix),result)
    plt.xlabel("Inclination (deg)")
    plt.ylabel("Maximum acceleration (m/s/s)")
    plt.title(title)



a = np.load("2acc_variance_alt1000_ra90.npy")
b = np.load("2acc_variance_alt500_ra90.npy")

plotacc(0,a,"Altitude 1000km, Deputy 1",'Maximum accceleration in s direction (m/s/s)')
plotacc(1,a,"Altitude 1000km, Deputy 2",'Maximum accceleration in s direction (m/s/s)')
plotacc(2,a,"Altitude 1000km, Deputy separation",'Maximum accceleration in b direction (m/s/s)')

plotacc(3,a,"Altitude 1000km, Deputy 1",'Delta_v in s direction (m/s)')
plotacc(4,a,"Altitude 1000km, Deputy 2",'Delta_v in s direction (m/s)')
plotacc(5,a,"Altitude 1000km, Deputy separation",'Delta_v in b direction (m/s)')

plotacc(6,b,"Altitude 500km, Deputy 1",'Maximum accceleration in s direction (m/s/s)')
plotacc(7,b,"Altitude 500km, Deputy 2",'Maximum accceleration in s direction (m/s/s)')
plotacc(8,b,"Altitude 500km,  Deputy separation",'Maximum accceleration in b direction (m/s/s)')


plotacc(9,b,"Altitude 500km, Deputy 1",'Delta_v in s direction (m/s)')
plotacc(10,b,"Altitude 500km, Deputy 2",'Delta_v in s direction (m/s)')
plotacc(11,b,"Altitude 500km,  Deputy separation",'Delta_v in b direction (m/s)')

print("Max acc 1000")
plot_max_ac(20,a,"Max Acc Hist: Altitude 1000km, Deputy 1, s direction")
plot_max_ac(21,a,"Max Acc Hist: Altitude 1000km, Deputy 2, s direction")
plot_max_ac(22,a,"Max Acc Hist: Altitude 1000km, Deputy separation, b direction")


print("delv 1000")
plot_ave_del_v(23,a,"Ave DelV Hist: Altitude 1000km, Deputy 1, s direction")
plot_ave_del_v(24,a,"Ave DelV Hist: Altitude 1000km, Deputy 2, s direction")
plot_ave_del_v(25,a,"Ave DelV Hist: Altitude 1000km, Deputy separation, b direction")


print("Max acc 500")
plot_max_ac(30,b,"Max Acc Hist: Altitude 500km, Deputy 1, s direction")
plot_max_ac(31,b,"Max Acc Hist: Altitude 500km, Deputy 2, s direction")
plot_max_ac(32,b,"Max Acc Hist: Altitude 500km, Deputy separation, b direction")

print("delv 500")
plot_ave_del_v(33,b,"Ave DelV Hist: Altitude 500km, Deputy 1, s direction")
plot_ave_del_v(34,b,"Ave DelV Hist: Altitude 500km, Deputy 2, s direction")
plot_ave_del_v(35,b,"Ave DelV Hist: Altitude 500km, Deputy separation, b direction")



plt.show()
