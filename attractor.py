#! /usr/bin/env python3

"""
Multi-dimensional continuous attractor network
Modeling non-linear systems.

Model:
    Array of neural units with fixed weightings. These fixed
        weightings are caculated to promote stronger connections
        to close neurons and smaller connections to distance neurons.
        In a weight matrix this looks like a diagonal that fades as
        it extends towards the corners of the matrix.
        This truth illustrates the properties of the weight matrix:
            Weight[x][x] > Weight[x][x-1] > Weight[x][x-2] and...
            Weight[x][x] > Weight[x][x+1] > Weight[x][x+2]
        - This is the attractor network of n nodes. Each node
            can be thought of as a vector representation within
            the phase space A of d-dimensionality where n>d.
        - The attractor network exists within a phase space A.
            This contains a set of network states that can be
            consider manifold-like. 
        - Attractor networks are initialized based on the input
            pattern. The dimensionality of the input pattern may
            differ from that of the network nodes. 
        - The trajectory of the network is a set of states along
            the evolution path as the network converges toward the
            attractor state. 
        - The basin of attraction is the set of states that results in
            movement towards a certain attractor. So in RatSLAM
            this basin of attraction is a phase line representing
            orientation 0-360 etc. 
    Neural unit (A single "continuous activation value"
                between 0 and 1 that is the activation 
                value of the unit. The weightings between
                the units doesnt change.
                It is fixed, only how the units activate
                changes.)
        - Calculate activity by summing the activity
            from other nerual units through the weighted
            connections.
        - Connections are weighted by their distance to eachother
        - Activation value increases when the system approaches
            a location associated with that neural unit. 
        - Can have many recurrent connections which
            will emphasise a state. Over time, without
            disturbance these self-stimulations will
            bring the system to a stable state. 
"""

import numpy as np

class ATTRACTOR(object):
    def __init__(self, ne=75, std=15, w=6):
        # FIXED VALUED PARAMS
        self.NE = ne # Number of excitory neruons
        self.dt = 0.001 # Time step

        self.tauE = 0.01 # time constant for excitory neurons
        self.gammaE = -1.5 # tonic inhibition of excitory neurons

        self.tauI = 0.002 # time constant for inhibitory neurons
        self.gammaI = -7.5 # tonic inhibition for inhibitor neurons
        
        self.std = std # intra-ring weighting field width (degrees)
        self.w = w # intra-ring weighting field strength

        self.WeightEI = -8.0    # E -> I weight scalar
        self.WeightIE = 0.880   # I -> E weight scalar
        self.WeightII = -4.0    # I -> I weight scalar

        # VARIABLES 
        self.E = np.zeros(self.NE) # NE number of excitory neurons
        self.I = 0                 # One inhibitor neuron
        self.W_EI = self.WeightEI * np.ones(self.NE)                    # E -> I Weights
        self.W_IE = self.WeightIE * np.ones(self.NE)                    # I -> I Weights
        self.W_II = self.WeightII                                       # I -> I Weights
        self.W_EE = self.build_weight_matrix(self.NE, self.std, self.w) # E -> E Weights

        # Recurrent variable
        self.lastE = self.E

    def build_weight_matrix(self, NE, std, w):
        # This process is kind of like building the attractor basin?
        variance = std**2 / (360**2) * NE**2
        i = np.ones((NE,1)) * np.arange(1,NE+1)
        j = np.arange(1,NE+1).reshape(NE,1) * np.ones((1,NE))
        d_choices = np.array((np.absolute(j + NE - i), np.absolute(i + NE - j), np.absolute(j - i)))
        d = np.amin(d_choices, axis=0)
        W = np.exp((-d * d)/variance)
        
        term = np.true_divide(W, (np.ones(NE) * np.sum(W, axis=0)))
        W = w * term
        return W

    def step(self, IN):
        self.lastE = self.E
        VE = self.W_EI * self.I + np.dot(self.W_EE, self.E) + self.gammaE + IN # Excitory compute)
        VI = self.W_II * self.I + np.dot(self.W_IE, self.E) + self.gammaI # Inhibitor compute 
        
        FE = self.tanh_activation(VE) # Excitory activation
        FI = self.tanh_activation(VI) # Inhibitor activation
        
        self.E = self.E + self.dt/self.tauE * (-self.E + FE) # Update excitory activity
        self.I = self.I + self.dt/self.tauI * (-self.I + FI) # Update inhibitor activity
        
        diff = np.sum(np.absolute(self.lastE - self.E)) / np.sum(self.E) 
        return self.E, diff

    def tanh_activation(self, x):
        return 0.5 + 0.5 * np.tanh(x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    net = ATTRACTOR()
    NN_w = net.W_EE
    plt.imshow(NN_w)
    plt.show()
    NE = net.NE
    #IN = np.concatenate((np.ones((100,1)) * np.random.random((1,NE)),np.zeros((100,NE))), axis=0)
    IN0 = 0.5 * (0.1 * np.random.random((1,NE)) + np.exp(-(np.power((np.arange(1,76) - 25),2))/40))
    IN1 = 0.5 * (0.1 * np.random.random((1,NE)) + np.exp(-(np.power((np.arange(1,76) - 45),2))/40))
    IN = np.ones((150,1)) * IN0 + IN1
    #IN = np.ones((10, 75))
    #print(np.shape(IN)[0])
    results = np.zeros((NE, np.shape(IN)[0]))
    print("TESTS: {}".format(np.shape(results)[1]))


    for put in range(len(IN)):
        E, diff = net.step(IN[put])
        #print("INPUT TEST {} [ERROR]:{}".format(put,np.around(diff, decimals=6)))
        results[:,put] = E
        plt.clf()
        plt.ylim((0,1))
        plt.stem(E, markerfmt='C0.')
        plt.pause(0.05)

    plt.stem(results[:,-1])
    plt.show()
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X,Y = np.mgrid[0:NE:1, 0:np.shape(IN)[0]:1]
    surf = ax.plot_surface(X, Y, results, cmap='afmhot')
    plt.show()
    """
