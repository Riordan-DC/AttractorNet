#! /usr/bin/env python3

from attractor import ATTRACTOR
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NE = 10
    #net = ATTRACTOR(NE, 13.4, 3.28)
    net = ATTRACTOR(NE, 10,2.4)
    
    E = np.zeros(NE)
   
    while True:
        spike_location = input("spike location :")
        try:
            spike_location = int(spike_location)
            print("Spike at {}".format(spike_location))
            E = np.zeros(NE)
            E[spike_location] = 1
        except:
            print("Attractor settling")
            while True:
                TH = 0.001
                difference = E
                E,_ = net.step(E)
                """
                plt.clf()
                plt.ylim((0,1))
                plt.stem(E, markerfmt='C0.')
                plt.pause(0.05)
                """
                change = np.sum(np.absolute(difference - E)) 
                print("[SAD of Excitations \ Strongest]:%2f \ %d" % (change, np.argmax(E)))
                if change < TH:
                    print("Attractor converged with threshold {}".format(TH))
                    break

        E,_ = net.step(E)
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.05)

        
