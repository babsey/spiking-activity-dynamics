import numpy as np
import pylab as pl

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# turn on the interactive mode for plotting
pl.ion()

# load the data
senders, ts = np.loadtxt('Result_Sim32_E120_I60-18001-0.gdf').T

# get the population size
npop = senders.max()

# estimate the average firing rate of the network
print ts.size / npop

# First step of visualizing data (2d plotting)
pl.plot(ts, senders,'.')
pl.show()

# 3d plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ts,senders % 120, senders // 120, '.')

# Focus on data in time window (snapshot)
idx = (ts > 100) * (ts <= 110)
tt, gg = ts[idx], senders[idx]
gx, gy = gg % 120, gg // 120
X = np.array([tt,gx,gy]).T

# Perform DBSCAN
db = DBSCAN(eps=1.4, min_samples=5)
clusters = db.fit(X)
labels = clusters.labels_

# check how many clusters I have
print np.unique(labels)

# 2d plotting
for i in range(-1, labels.max()+1):
    aa = labels == i
    pl.plot(tt[aa], gg[aa], '.')

# 3d plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(-1, labels.max()+1):
    aa = labels == i
    ax.plot(tt[aa], gx[aa], gy[aa], '.')
