import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#helio60 = pd.read_json("bigboi_helio60.json","records")
helio40 = pd.read_json("hopeful_H40.json","records")
#NZ60 = pd.read_json("bigboi_NZ60.json","records")
#NZ40 = pd.read_json("bigboi_NZ40.json","records")

colourmap = 'inferno'

A_h60 = np.zeros((90,180))
A_h40 = np.zeros((90,180))
A_N60 = np.zeros((90,180))
A_N40 = np.zeros((90,180))

for i in range(180):
    for j in range(90):
        print(i,j)
        #A_h60[j,i] = helio60.iloc[i*90 + j]["percent"]
        A_h40[j,i] = helio40.iloc[i*90 + j]["percent"]
        #A_N60[j,i] = NZ60.iloc[i*90 + j]["percent"]
        #A_N40[j,i] = NZ40.iloc[i*90 + j]["percent"]
"""
plt.figure(1)
plt.clf()
plt.imshow(A_h60,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
cbar = plt.colorbar()
cbar.set_label('Observability (%)', rotation=270, labelpad=15)
plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.title("Full Year Observability for \n Heliosynchronous Orbit, 60deg Anti-Sun")
#plt.savefig("Obs_plots/full_h60.png")
"""
plt.figure(2)
plt.clf()
plt.imshow(A_h40,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
cbar = plt.colorbar()
cbar.set_label('Observability (%)', rotation=270, labelpad=15)
plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.title("Full Year Observability for \n Heliosynchronous Orbit, 40deg Anti-Sun")
plt.savefig("Obs_plots/full_h40.png")

"""
plt.figure(3)
plt.clf()
plt.imshow(A_N60,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
cbar = plt.colorbar()
cbar.set_label('Observability (%)', rotation=270, labelpad=15)
plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.title("Full Year Observability for \n 39deg Orbit, 60deg Anti-Sun")
plt.savefig("Obs_plots/full_N60.png")

plt.figure(4)
plt.clf()
plt.imshow(A_N40,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
cbar = plt.colorbar()
cbar.set_label('Observability (%)', rotation=270, labelpad=15)
plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.title("Full Year Observability for \n 39deg Orbit, 40deg Anti-Sun")
plt.savefig("Obs_plots/full_N40.png")
"""

for month in range(1,13):
    #A_h60 = np.zeros((90,180))
    A_h40 = np.zeros((90,180))
    #A_N60 = np.zeros((90,180))
    #A_N40 = np.zeros((90,180))
    
    for i in range(180):
        for j in range(90):
            print(i,j)
            #A_h60[j,i] = helio60.iloc[i*90 + j]["Calendar"][month-1]
            A_h40[j,i] = helio40.iloc[i*90 + j]["Calendar"][month-1]
            #A_N60[j,i] = NZ60.iloc[i*90 + j]["Calendar"][month-1]
            #A_N40[j,i] = NZ40.iloc[i*90 + j]["Calendar"][month-1]
    """
    plt.figure(1)
    plt.clf()
    plt.imshow(A_h60,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
    cbar = plt.colorbar()
    cbar.set_label('Observability (%)', rotation=270, labelpad=15)
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    plt.title("Month %s Observability for \n Heliosynchronous Orbit, 60deg Anti-Sun"%month)
    plt.savefig("Obs_plots/h60/month%s_h60.png"%month)
    """
    plt.figure(2)
    plt.clf()
    plt.imshow(A_h40,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
    cbar = plt.colorbar()
    plt.clim(0,100)
    cbar.set_label('Observability (%)', rotation=270, labelpad=15)
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    plt.title("Month %s Observability for \n Heliosynchronous Orbit, 40deg Anti-Sun"%month)
    plt.savefig("Obs_plots/h40/month%s_h40.png"%month)
    """
    plt.figure(3)
    plt.clf()
    plt.imshow(A_N60,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
    cbar = plt.colorbar()
    cbar.set_label('Observability (%)', rotation=270, labelpad=15)
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    plt.title("Month %s Observability for \n 39deg Orbit, 60deg Anti-Sun"%month)
    plt.savefig("Obs_plots/N60/month%s_N60.png"%month)
    
    plt.figure(4)
    plt.clf()
    plt.imshow(A_N40,origin="lower",cmap=colourmap,extent=[-180,180,-90,90])
    cbar = plt.colorbar()
    cbar.set_label('Observability (%)', rotation=270, labelpad=15)
    plt.xlabel("Right Ascension (deg)")
    plt.ylabel("Declination (deg)")
    plt.title("Month %s Observability for \n 39deg Orbit, 40deg Anti-Sun"%month)
    plt.savefig("Obs_plots/N40/month%s_N40.png"%month)
    """