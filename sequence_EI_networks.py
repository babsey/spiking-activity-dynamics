# -*- coding: utf-8 -*-
#
# sequence_EI_networks.py
#
# Copyright 2017 Sebastian Spreizer
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

now = datetime.datetime.now()
now_str = now.strftime('%Y%m%d-%H%M%S')
output_file = 'EI_networks_sequence_%s' %now_str

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
Create nodes
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
    "V_th": -55.0
}

nrowE, ncolE = 120, 120
nrowI, ncolI = 60, 60

npopE = nrowE * ncolE
npopI = nrowI * ncolI

popE = nest.Create("iaf_psc_alpha", npopE,
        params = neuron_params)
popI = nest.Create("iaf_psc_alpha", npopI,
        params = neuron_params)
pop = popE + popI

# Create input devices
noiseE = nest.Create('noise_generator')
noiseI = nest.Create('noise_generator')
noise = noiseE + noiseI

# Create recording devices
sd = nest.Create('spike_detector', params={
    'start':        500.0,
    'to_file':      True,
    'label':        output_file,
})


"""
Connect nodes
"""

landscape = cl.Perlin_uniform(nrowE, {'size': 4})
move = cl.move(nrowE)
offsetE = popE[0]
offsetI = popI[0]

p = 0.05
stdE = 12
stdI = 9
shift = 1
Jx = 10.0
g = 6

syn_specE = {'weight': Jx}
syn_specI = {'weight': g * -Jx}
for idx in range(npopE):

    # E-> E
    source = idx, nrowE, ncolE, nrowE, ncolE, int(p * npopE), stdE
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    if landscape is not None:        #  asymmetry
        targets = (targets + shift * move[landscape[idx] % len(move)]) % npopE
    targets = targets[targets != idx]
    nest.Connect([popE[idx]], (targets + offsetE).tolist(), syn_spec=syn_specE)

    # E-> I
    source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popE[idx]], (targets + offsetI).tolist(), syn_spec=syn_specE)

for idx in range(npopI):
    kwargs = {'syn_spec': {'weight': g * -Jx}}

    # I-> E
    source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    nest.Connect([popI[idx]], (targets + offsetE).tolist(), syn_spec=syn_specI)

    # I-> I
    source = idx, nrowI, ncolI, nrowI, ncolI, int(p * npopI), stdI
    targets, delay = lcrn.lcrn_gauss_targets(*source)
    targets = targets[targets != idx]
    nest.Connect([popI[idx]], (targets + offsetI).tolist(), syn_spec=syn_specI)

# Connect noise input device to all neurons
nest.Connect(noiseE, popE)
nest.Connect(noiseI, popI)

# Connect spike detector to population of all neurons
nest.Connect(pop, sd)


"""
Warming up
"""

nest.SetStatus(noise, params={
        "std": 500.0
    })
nest.Simulate(250.)
nest.SetStatus(noise, params={
        "mean": 300.0,
        "std": 50.0
    })
nest.Simulate(250.)


"""
Start simulation
"""

nest.Simulate(1000.)


"""
Plot spiking activity
"""

sdE = nest.GetStatus(sd, 'events')[0]
ts,gids = sdE['times'], sdE['senders']
fig,ax = pl.subplots(1)
ax.plot(ts, gids, '|')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Neuron')
fig.savefig('./Data/' + output_file + '.png')
