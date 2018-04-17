# -*- coding: utf-8 -*-
#
# sequence_EI_networks.py
#
# Copyright 2018 Sebastian Spreizer
# The MIT License

"""
Script for NEST simulation of EI network models to produce activity sequences.
"""

import sys
import numpy as np
import nest
import pylab as pl
import datetime

import lib.lcrn_network as lcrn
import lib.connectivity_landscape as cl
import lib.animation as anim

now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d-%H%M%S')
output_file = 'EI_networks_sequence_%s' % now_str

"""
Set Kernel Status
"""

np.random.seed(0)
nest.ResetKernel()
nest.SetKernelStatus({
    'local_num_threads': 4,
    'resolution': 0.1,
    'data_path': './Data',
    'overwrite_files': True,
})


"""
Create neurons
"""

neuron_params = {
    "C_m": 250.0,
    "E_L": -70.0,
    "t_ref": 2.0,
    "tau_m": 10.0,
    "tau_minus": 20.0,
    "tau_syn_ex": 5.0,
    "tau_syn_in": 5.0,
    "V_reset": -70.0,
    "V_th": -55.0,
}

nrowE, ncolE = 120, 120
nrowI, ncolI = 60, 60

npopE = nrowE * ncolE
npopI = nrowI * ncolI

popE = nest.Create("iaf_psc_alpha", npopE,
                   params=neuron_params)
popI = nest.Create("iaf_psc_alpha", npopI,
                   params=neuron_params)
pop = popE + popI


"""
Distribute V_m
"""
V_m = np.random.normal(-65., 5., len(pop))
for idx in range(len(pop)):
    nest.SetStatus([pop[idx]], {'V_m': V_m[idx]})


"""
Create devices
"""

noise_params = {
    'mean': 300.,               # 300. - 600.
    'std': 50.,                 # 0. - 300.
}

# Create input devices
noiseE = nest.Create('noise_generator', params=noise_params)
noiseI = nest.Create('noise_generator', params=noise_params)
noise = noiseE + noiseI

# Create recording devices
sd = nest.Create('spike_detector', params={
    'to_file':      True,
    'label':        output_file,
})


"""
Get spatial connection landscape
"""

# landscape = cl.random(nrowE, {'seed': 0})
# landscape = cl.homogeneous(nrowE, {'phi': 3})
# landscape = cl.Perlin(nrowE, {'size': 4})
landscape = cl.Perlin_uniform(nrowE, {'size': 4, 'base': 0})


"""
Connect neurons
"""

move = cl.move(nrowE)
offsetE = popE[0]
offsetI = popI[0]

p = 0.05                    # 0.05 - 0.1
stdE = 12
stdI = 9                    # 9 - 11
shift = 1                   # 1 - 3
Jx = 10.0
g = 8                       # 4 - 8


syn_specE = {'weight': Jx}
for idx in range(npopE):

    # E-> E
    source = idx, nrowE, ncolE, nrowE, ncolE, int(p * npopE), stdE
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    if landscape is not None:  # asymmetry
        targets = (targets + shift * move[landscape[idx] % len(move)]) % npopE
    targets = targets[targets != idx]
    nest.Connect([popE[idx]], (targets + offsetE).tolist(), syn_spec=syn_specE)

    # E-> I
    source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popE[idx]], (targets + offsetI).tolist(), syn_spec=syn_specE)

syn_specI = {'weight': g * -Jx}
for idx in range(npopI):

    # I-> E
    source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popI[idx]], (targets + offsetE).tolist(), syn_spec=syn_specI)

    # I-> I
    source = idx, nrowI, ncolI, nrowI, ncolI, int(p * npopI), stdI
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    targets = targets[targets != idx]
    nest.Connect([popI[idx]], (targets + offsetI).tolist(), syn_spec=syn_specI)


"""
Connect devices to neurons
"""

# Connect noise input device to all neurons
nest.Connect(noiseE, popE)
nest.Connect(noiseI, popI)

# Connect spike detector to population of all neurons
nest.Connect(pop, sd)


"""
Start simulation
"""

wuptime = 100.
nest.Simulate(wuptime)

simtime = 2000.
nest.Simulate(simtime)


"""
Get data from memory
"""

sdE = nest.GetStatus(sd, 'events')[0]
ts, gids = sdE['times'], sdE['senders']

idx = ts > wuptime
ts, gids = ts[idx]-wuptime, gids[idx]


"""
Get data from file
"""

# sd_id = 18003
# output_file = 'EI_networks_sequence_20180417-091947'
# data = []
# for i in range(4):
#     d = np.loadtxt('./Data/%s-%s-%s.gdf' % (output_file, sd[0], i))
#     data.extend(d)
# gids, ts = np.array(data).T


"""
Sort data by time, important for ISI
"""
idx = np.argsort(ts)
gids, ts = gids[idx], ts[idx]


"""
Split data in two populations
"""

gidxE = gids - offsetE < npopE
tsE, gidsE = ts[gidxE], gids[gidxE]         # Excitatory population
tsI, gidsI = ts[~gidxE], gids[~gidxE]       # Inhibitory population


"""
Plot spiking activity
"""

fig, ax = pl.subplots(1)
ax.plot(tsE, gidsE, '|')
ax.plot(tsI, gidsI, '|')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
fig.savefig('./plots/%s.png' % (output_file))


"""
Animate spike activity
"""


ts_bins = np.arange(0., simtime + 1, 20.)
h = np.histogram2d(tsE, gidsE - offsetE, bins=[ts_bins, range(npopE + 1)])[0]
hh = h.reshape(-1, nrowE, ncolE)

fig, ax = pl.subplots(1)
a = anim.images(ax, hh, vmin=0, vmax=np.max(hh))
a.save('./plots/%s.mp4' % (output_file), fps=10,
       extra_args=['-vcodec', 'libx264'])


"""
Show the figures
"""

pl.show()
