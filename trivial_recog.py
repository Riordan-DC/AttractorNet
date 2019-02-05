#! /usr/bin/env python3

from attractor import ATTRACTOR
import numpy as np
import sys
import matplotlib.pyplot as plt 

NE = 30
std = float(sys.argv[1])
w = float(sys.argv[2])
net = ATTRACTOR(NE, std, w)

# Input model
"""
stimulation:     s1  s2          s3
TIME:       |----^---^-----------^--->

Idea:
    Associate each reference image with a time cell.
    This cell is a local inhibitory neuron that only inhibits
    nurons before it, not after it. This encourages the generation
    and stimulation of new local view cells.

Idea 2:
    Subidivde networks into smaller more stable attractor basins.
    Essentially, not a fully connection network, more like a network
    with overlapping bounds.
"""

# Settle activity
for i in range(100):
    inpt = np.zeros(NE)
    inpt[15] = 1
    E,_ = net.step(inpt)

# Pump stimulation over forward cycles
def pump(target, distance, cycles):
    sub_packet = np.zeros((distance, NE))
    for stimulation in range(distance):
        sub_packet[stimulation, target+stimulation] = 1
    packet = np.tile(sub_packet, (cycles, 1))
    return packet

def calibrate():
    avg_iters = np.zeros(NE)
    for stim in range(NE):
        # Build packet
        packet = np.zeros(NE)
        packet[stim] = 1

        E,_ = net.step(packet)
        # Step net
        iters = 0
        while np.argmax(E) != stim:
            iters += 1
            E,_ = net.step(packet)

        avg_iters[stim] = iters

    return np.average(avg_iters)

impulse_avg = calibrate()
impulse_scale = 0.5

responses = np.zeros((NE,NE))

for stim in range(NE):
    # Build packet
    packet = np.zeros(NE)
    packet[stim] = 1

    E,_ = net.step(packet)
    # Step net
    for i in range(int(impulse_avg * impulse_scale)):
        E,_ = net.step(packet)
        responses[stim] = E
    
    #E,_ = net.step(np.zeros(NE))
   
    # Draw stimulations 
    plt.clf()
    plt.ylim((0,1))
    plt.stem(E)
    plt.pause(0.05)
