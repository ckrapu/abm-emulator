# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:10:52 2021

Metabolic Stability and Epigenesis in Randomly Constructed Genetic Nets by Stuart 
Kauffman, Journal of Theoretical Biology (1969) vol 22, 437-467


Random (uniform) values of p and K for each iteration.

Take mean over all agent states at T = 20.

Use GP regression to infer response between (p,K) and < x_20 >

@author: bruce
"""



import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import matplotlib
import matplotlib.pyplot as plt

import time

start_time = time.time()

PLOT_GATE = True

######################################
######################################
# fixed population parameters
######################################

N = 1000 # number of "genes" or agents or nodes
T = 20#time steps

# free parameters ######################################

# two parameters: k and p
# chosen uniformly at random from following intervals
K = [1, 16] # [K_min, K_max]
# p uniform on [0,1)

# expected number of 1's in initial conditions
# x_0 random, defined inside parameter loop

# number of iterations / samples
nIter = 30
#####################################

#Got seed ? 
#np.random.seed(718281828)

#####################################################
# ABM function
######################################################

def kaufNetABM(N,x_0,K,p,T):
    stateMean = np.array([])
    #choose influencers for each node
    edgeList = []
    for agent in range(N):
        k = K
        #k = np.random.poisson(K-1)+1
        edgeList = edgeList + [np.random.choice(N,k,replace=False)]
        
    #state vector
    X = np.zeros((T+1,N))
    #set initial conditions
    initialConditions = np.random.binomial(1,x_0,N)
    X[0,] = initialConditions   
        
    stateMean = np.append(stateMean,X[0,].mean())
        
    #random logic for each agent
    # each agent assigned a random map, agentMap (dictionary)
    # maps are indexed in a (list), mapList
    # there are 2**K possible inputs, and a p-coin is flipped to determine the value at each input
    mapList = []
    for agent in range(N):
        agentMap = {}
        for key in range(2**K):
            agentMap[key] = np.random.binomial(1,p)
        mapList = mapList + [agentMap]
        
    #dynamics
    for t in range(T):
        for agent in range(N):
       #base 2 representation of inputs, to match encodeing in dictionary 
            mapKey = 0
            for k in range(K):     
                mapKey = mapKey + (2**k)*X[t,edgeList[agent][k]]
            X[t+1,agent] = mapList[agent][mapKey]
        
        #record stats
        stateMean = np.append(stateMean, X[t+1,].mean())
    return(stateMean[T])

######################################################### end ABM function



# Data ####################
output = []
###############################


################################
# Iterating ABM over parameters
################################


for iter in range(nIter):
    x_0 = np.random.rand()
    k = np.random.randint(K[0],K[1])
    p = np.random.rand()
    stateMean = kaufNetABM(N,x_0,k,p,T)
    output.append([p,k,stateMean])

            
print("---  ABM sims: %s seconds ---" % (time.time() - start_time))


#############################################################
#%% GP training, scikitlearn
next_time = time.time()

output = np.array(output)
#input into GP model .fit()
# shape = number samples x number parameters 
trainingParams = output[:,:2]
abmOutput = output[:,2]

# # # # # # # # # # # # # # # # # # # # #
#Instantiate a GP model
kernel = ConstantKernel(constant_value = 0.5, constant_value_bounds = (0.1,4))*RBF(length_scale = 0.5,length_scale_bounds =(1e-2,5)) 
#kernel = ConstantKernel(0.5, (1e-2,1))+RBF(2,(1e-1,10))
gp = GaussianProcessRegressor(kernel = kernel,n_restarts_optimizer = 10, alpha = 0.01,normalize_y = True)


#Fit to data using MLE of the parameters
gp.fit(trainingParams,abmOutput)

# # Kp grid points
nPred = 20
pPlot = np.linspace(0.05,0.95,num = nPred)
kPlot = np.linspace(1,15,num = nPred)

predParams = np.zeros((nPred**2,2))
count = 0
for p in range(nPred):
    for k in range(nPred):
        predParams[count,] = [pPlot[p],kPlot[k]]
        count += 1


# #make the prediction
y_pred, sigma = gp.predict(predParams,return_std = True)

print("---SKL Emulator: %s seconds ---" % (time.time()-next_time))



#----------------------------------------------
if PLOT_GATE:
    plotData = np.reshape(y_pred,(nPred,nPred))
    
    fig, ax = plt.subplots()
    im = ax.imshow(plotData,aspect = 'auto',extent = [kPlot[0],kPlot[nPred-1],pPlot[0], pPlot[nPred-1]],origin = 'lower',vmin = 0, vmax = 1)
    plt.plot(trainingParams[:,1],trainingParams[:,0],'rx')
    plt.colorbar(im)
    plt.xlabel('K')
    plt.ylabel('p')
    
    # plt.figure()
    # plt.plot(trainingParams[0:T+1,1],dataPlot[nIter,],'rx')
    # plt.plot(t_pred,y_pred,'k:')
    # plt.ylim((0,1))
    # plt.fill(np.concatenate([t_pred,t_pred[::-1]]),np.concatenate([y_pred-1.9600*sigma,(y_pred+1.9600*sigma)[::-1]]), alpha = .5,fc = 'b',ec='None')
    # plt.xlabel('Time')
    # plt.ylabel('Mean State Value')
#----------------------------------------------
