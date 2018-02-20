# -*- coding: utf-8 -*-
#
# simulate_random_networks.py
#
# Copyright 2018 Sebastian Spreizer
# The MIT License

import numpy as np
import nest
import pylab as pl

import lib.local_values_landscape as lvl


"""
Set Kernel Status
"""

np.random.seed(0)
nest.ResetKernel()
nest.SetKernelStatus({
    'resolution': 0.1,
})


"""
Create nodes
"""

nrow, ncol = 100, 100
npop = nrow * ncol
pop = nest.Create('iaf_psc_alpha', npop)

nest.SetStatus(pop, 'V_m', np.random.normal(-70.0, 5., npop).tolist())

V_th = (lvl.Perlin(nrow) * 5.) - 50.
nest.SetStatus(pop, 'V_th', V_th.tolist())

tau_m = (lvl.Perlin(nrow) * 5.) + 5.
nest.SetStatus(pop, 'tau_m', tau_m.tolist())

# Create input devices
pg = nest.Create('poisson_generator', params={
    'rate': 10000.
})

GWN = nest.Create('noise_generator', params={
    'mean': 1000.,
    'std': 500.,
})

# Create recording devices
sd = nest.Create('spike_detector')


"""
Connect nodes
"""

# Connect input device to all neurons
nest.Connect(GWN, pop)
# nest.Connect(pg, pop)

# Recurrent connection
nest.Connect(pop, pop, conn_spec={
    'rule': 'fixed_outdegree',
    'outdegree': 1000
}, syn_spec={
    'weight': -10.
})

# Connect spike detector to all neurons
nest.Connect(pop, sd)


"""
Start simulation
"""

nest.Simulate(1000.)


"""
Plot spiking activity
"""

sdE = nest.GetStatus(sd, 'events')[0]
ts, gids = sdE['times'], sdE['senders']

fig, ax = pl.subplots(1)
im = ax.matshow(V_th.reshape(nrow, ncol))
ax.set_xlabel('Column')
ax.set_ylabel('Row')
cbar = pl.colorbar(im)
cbar.set_label('Spike threshold')

fig, ax = pl.subplots(1)
ax.plot(ts, gids, '|')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')

fig, ax = pl.subplots(1)
h = np.histogram(gids - 1, bins=range(npop + 1))[0]
im = ax.matshow(h.reshape(nrow, ncol))
ax.set_xlabel('Column')
ax.set_ylabel('Row')
cbar = pl.colorbar(im)
cbar.set_label('Spike count')

fig, ax = pl.subplots(1)
ax.plot(V_th, h, '.')
ax.set_xlabel('Spike threshold')
ax.set_ylabel('Spike count')

pl.show()
