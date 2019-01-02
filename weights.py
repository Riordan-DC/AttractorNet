#!/usr/bin/env python3


import numpy as np

"""
    A weight matrix expresses the weights between neurons.
    A complete matrix for a fully connected network (as in,
    all neurons are connection to all others) must contain
    a weight for every possible combination of neuron.
    These helper functions build these combinations and store
    them in a matrix. 

    Computationally these are store as a matrix of weights
    for each neuron.
"""

def buildWeights2D(NE, std, w):
    '''
    Builds a weight matrix for an attractor network with 2 dimensions.
    Commonly these dimensions are Yaw and Pitch. 
    
    Think of the concept as having to build a weight matrix for each
    neuron in a set of N neurons. 

    Note: Dont forget that both dimensions are the same length.

    If we want to compare each neuron agaisnt each other neuron we
    need to loop N**N times. This is achieved in a nested for-loop.
    
    let N = length of both dimensions
    For each neuron in N_D1:
        For each neuron in N_D2:
            Calculate weight matrix -> (N x N)

    Now for just Neuron 1 we have created N x (N x N) weight matracies
    This is repeated for all N neurons. Giving us: (N x (N x N)) x N
    
    
    Because this is an ordered set of weights {x,y} != {y,x}
   
    Let x and y be an index within dimension 1 and 2 respectively.
    We are building a weight matrix where:
        W_MATRIX[:,:,x,y] = W_XY(N * N) ->

             X   X   X 
        --------------
        Y_1|(xy)(xy)(xy) 
        Y_2|(xy)(xy)(xy)
        Y_3|(xy)(xy)(xy)
    '''
    Weights = np.zeros((NE,NE,NE,NE))
    variance = np.power(std, 2)
    [j, i] = np.meshgrid(np.arange(0,NE), np.arange(0,NE))
    dimension_length = np.arange(0,NE)
    for ii in dimension_length:
        for jj in dimension_length:
            dx = ii - i
            dy = jj - j
            min_term_x = np.min(np.array((np.absolute(dx),
                    np.absolute(NE + dx),
                    np.absolute(NE - dx))), 2) #Missing empty array.
                    #If there is a problem please check the matlab code
            mindx = np.true_divide(min_term_x, NE)
            min_term_y = np.min(np.array((np.absolute(dy),
                    np.absolute(NE + dy),
                    np.absolute(NE - dy))), 2) #Missing empty array.
                    #If there is a problem please check the matlab code
            mindy = np.true_divide(min_term_y, NE)
            d2 = np.power(mindx,2) + np.power(mindy,2)
            Weights[:,:,ii,jj] = d2
    print(Weights)
    Weights = np.true_divide(w, (2 * np.pi * variance) * np.exp(np.true_divide(np.true_divide(-Weights, 2), variance)))
    return Weights
            

def buildWeights3D(NE, std, w):
    pass

if __name__ == "__main__":
    print("These are helper functions for building weight\n matricies for attractor networks with multiple\n dimensions")
    buildWeights2D(3,15,6)
