#! /usr/bin/env python3

import numpy as np
import cv2

def build_weight_matrix(NE, std, w):
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

if __name__ == "__main__":
    std = 1
    while True:
        W = build_weight_matrix(100,std,6)
        std += 1
        cv2.imshow('weight', W)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
