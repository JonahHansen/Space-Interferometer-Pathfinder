import numpy as np
import matplotlib.pyplot as plt

np40 = np.load("inch_dec_40.npy")
np60 = np.load("inch_dec_60.npy")
total = np40 + np60
plt.figure(1)
plt.imshow(total[:,0,:].transpose(),aspect="auto",extent=[0,np.shape(total)[0]/30/24,-90,90])
plt.xlabel("Time (days)")
plt.ylabel("Dec (degrees)")

np40 = np.load("inch_ra_40.npy")
np60 = np.load("inch_ra_60.npy")
total = np40 + np60
plt.figure(2)
plt.imshow(total[:,:,0].transpose(),aspect="auto",extent=[0,np.shape(total)[0]/15/24,0,360])
plt.xlabel("Time (days)")
plt.ylabel("Ra (degrees)")
