# -*- coding: utf-8 -*-
#
# plot_connection_matrix_EI_networks.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import matplotlib as mpl
import pylab as pl

import lib.connection_matrix as cm
import lib.distance as dist

landscapes = [
    # {'mode': 'sy/mmetric'},
    # {'mode': 'random'},
    # {'mode': 'Perlin', 'specs': {'size': 4}},
    # {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

nrowE, ncolE = 120, 120
nrowI, ncolI = 60, 60
p = 0.05                    # connection probability
stdE = 12                   # space constant for E targets
stdI = 6                    # space constant for I targets

nmax = 150
W_bins = np.arange(0, nmax)

for landscape in landscapes:
    W = cm.EI_networks(landscape, nrowE, ncolE, nrowI, ncolI, p, stdE, stdI)
    hist_W = []
    for Widx, Wi in enumerate(W):
        hist_Wi = np.array([np.histogram(Wii, bins=W_bins)[0] for Wii in Wi])
        hist_W.append(hist_Wi)


labels = ['E -> E', 'E -> I', 'I -> E', 'I -> I']
fig, ax = pl.subplots(1)
for Widx, hist_Wi in enumerate(hist_W):
    ax.plot(W_bins[:-1], np.mean(hist_Wi,0), label=labels[Widx])

ax.set_xlabel('Number of connections')
ax.set_ylabel('Amount')
ax.legend()

def plot_connections(Wi, nrow, pixel_dist):
    center = nrow * nrow / 2 + nrow / 2
    Wrolled = np.array([np.roll(x, center-i) for i,x in enumerate(Wi[:n])])

    fig,ax = pl.subplots(1)
    norm = mpl.colors.LogNorm(vmin=0.1, vmax=40)
    im = ax.matshow(np.mean(Wrolled,0).reshape(nrow, nrow), norm=norm)
    cbar = fig.colorbar(im)
    cbar.set_label('Averaged number of connections')
    ax.set_xlabel('Row')
    ax.set_ylabel('Col')

    d = dist.distance_matrix(nrow,nrow) * pixel_dist
    ax1.plot(d, np.mean(Wrolled,0), '.')
    ax1.set_xlabel('Distance from source neuron')
    ax1.set_ylabel('Averaged number of connections')

n = 1000
pixel_dist = [1, 2]
nrowX = [nrowE, nrowI]

fig, ax1 = pl.subplots(1)
for idx, Wi in enumerate([W[0],W[-1]]):
    plot_connections(Wi, nrowX[idx], pixel_dist[idx])

pl.show()
