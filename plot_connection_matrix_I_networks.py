# -*- coding: utf-8 -*-
#
# plot_connection_matrix_I_networks.py
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
ncon = 1000             # number of connections
kappa = 4               # shape
theta = 3               # scale


a = []
for idx, landscape in enumerate(landscapes):
    W = cm.I_networks(landscape, nrow, ncol, ncon, kappa, theta)
    a.append(W)

for idx, c in enumerate(a):
    pl.matshow(c)
    pl.title(landscape[idx]['mode'])

pl.show()
