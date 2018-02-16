# -*- coding: utf-8 -*-
#
# plot_eigenvalue_I_networks.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import pylab as pl

import lib.connection_matrix as cm

landscapes = [
    {'mode': 'symmetric'},
    {'mode': 'random'},
    {'mode': 'Perlin', 'specs': {'size': 4}},
    {'mode': 'Perlin_uniform', 'specs': {'size': 4}},
    {'mode': 'homogeneous', 'specs': {'phi': 3}},
]

nrow, ncol = 100, 100
npop = nrow * ncol
ncon = 1000
kappa = 4               # shape
theta = 3               # scale
n = 1000


a = []
for idx, landscape in enumerate(landscapes):
    W = cm.I_networks(landscape, nrow, ncol, ncon, kappa, theta)
    a.append(W)


fig, ax = pl.subplots(1)
for idx, c in enumerate(a):
    w, v = np.linalg.eig(-1. * c[:n, :n])
    ax.plot(w.real, w.imag, 'o', label=landscapes[idx]['mode'])

ax.set_title('Eigenvalue')
ax.set_xlabel('Real part')
ax.set_ylabel('Imag part')

pl.legend()
pl.show()
