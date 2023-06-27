#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 08:17:20 2019

@author: barnstedto

=> Deconvolution with Autoregressive Model 

"""
import numpy as np
from tqdm import tqdm


def Kalman_Filt_v(pos,dt) :
    measurements = pos;
    
    ## initialize
    #    x  x' 
    x = np.array([[pos[0]], [0]], ndmin=2)      		#Initial State (Location and velocity and acceleration)
   
    P = np.array([[1, 0], [0, 100]])       		# Initial Uncertainty
    
    A = np.array([[1, dt], [0, 1]])# Transition Matrix
    
    # Measurement function
    H = np.array([1, 0], ndmin=2)
    
    # measurement noise covariance; 1e-3 recommended, smaller values for precise onset, larger for smooth velocity
    R = 1e-3
    
    # Process Noise Covariance
    Q = np.array([[1/4*dt**4, 1/2*dt**3], [1/2*dt**3, dt**2]])
    
    # Identity matrix
    I = np.identity(2)

    

    ## compare to datapoints
    posF = [None] * len(measurements)
    vF = [None] * len(measurements)
    
    with tqdm(total=len(measurements)) as pbar:
        for n, measurement in enumerate(measurements):
    
            # Prediction
            x = np.matmul(A, x)                	# predicted State
            P = A @ P @ A.transpose() + Q             # predicted Covariance
    
            # Correction
            Z = measurement
            y = Z - np.matmul(H, x)              # Innovation from prediction and measurement
            S = H @ P @ H.transpose() + R          # Innovation-covariance
            K = np.matmul(P, H.transpose()) / S          # Filter-Matrix (Kalman-Gain)
    
            x = x + (K * y)              # recalculation of system state
            #print(x)
            posF[n] = np.float64(x[0])
            vF[n] = np.float64(x[1])
            
            P = np.matmul(I - (np.matmul(K, H)), P)          # recalculation of covariance
            pbar.update(1)
            
    return posF, vF