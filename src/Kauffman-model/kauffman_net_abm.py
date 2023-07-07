# -*- coding: utf-8 -*-
"""

Metabolic Stability and Epigenesis in Randomly Constructed Genetic Nets by Stuart 
Kauffman, Journal of Theoretical Biology (1969) vol 22, 437-467


Iterating over a set of values for pairs (p,K)
Simulating "experimental data"
ver 3a --- fix network topology over each simulation
ver 3b --- return entire trajectory, X -  Xb

??? Networks resampled each iteration, with new K certainly.
Here, K is expected degree, and thus a continuous parameter

Each node is given a random function. p is (expected) fraction of '1's in the output
function of each agent.

See "Boolean Dynamics with Random Couplings" by Aldana, Coppersmith, and Kadanoff
29 APR 2002 arXiv:nlin/0204062v2
 info on hamming distance and divergence of orbits (section 2.1.1) and
 characterization via noise (section 5.2)
 
 figure 2.4 Kp phase diagram
 figure 2.1 normalized hamming distance D(t) for different values of p
 

save results as a data frame (panda) and write to csv to be assimilated in pymc3

@author: bruce
"""



import numpy as np
import pandas as pd 

import time



######################################
######################################
#%% Model parameters
######################################

N = 1000 # number of "genes" or agents or nodes
T = 40#time steps

# free parameters ######################################

# two parameters: k and p
K = 4 #
P = [0.05]#, 0.2, 0.4] # uniform on [0,1)
N_population = len(P)

# expected number of 1's in initial conditions
# x_0 random, defined inside parameter loop

# number of iterations / samples
nIter = 1
#####################################

#Got seed ? 
#np.random.seed(718281828)

#%% Helper functions
def HamDist(x,y):
    return(sum((x-y)**2))

def MakeGraph(N,K):
    #choose influencers for each node
    #Construct Graph
    edgeList = []
    edge_prob = K / N
    degreeList = []
    for agent in range(N):
        # choose how many neighbors each agent will have
        agent_degree = np.random.binomial(N,edge_prob)
        degreeList = degreeList + [agent_degree]
        # choose exactly which neighbors they are
        edgeList = edgeList + [np.random.choice(N,agent_degree,replace=False)]
    
    return(edgeList,degreeList)


      
#####################################################
#%% ABM function
######################################################

def kaufNetABM(N,x_0,p,T,edgeList,degreeList):
  
    # function output:
    hamDist = np.zeros(T)
    #state vector
    X = np.zeros((T+1,N))
    #set initial conditions
    initialConditions = np.random.binomial(1,x_0,N)
    X[0,] = initialConditions 
    #Perturbation
    # random reassignment of each element of X, iid p = 0.05 (Np = expected number of perturbed sites)
    Xb = X.copy()
    Xb[0,] = (Xb[0,] + np.random.binomial(1,0.01,N))%2
        
    #random logic for each agent
    # each agent assigned a random map, agentMap (dictionary)
    # maps are indexed in a (list), mapList
    # there are 2**K possible inputs, and a p-coin is flipped to determine the value at each input
    # here K is the (in)degree of each agent, degreeList[agent]
    mapList = []
    # variables for measuring the p-value generated from this specific sample
  #  totalValues = 0
  #  sumTotal = 0
    for agent in range(N):
        agentMap = {}
        coinflipArray = np.random.binomial(1,p,2**degreeList[agent])
   #     sumTotal += sum(coinflipArray)
    #    totalValues += len(coinflipArray)
        for key in range(2**degreeList[agent]):
            agentMap[key] = coinflipArray[key]
        mapList = mapList + [agentMap]
        
    #dynamics
    for t in range(T):
        for agent in range(N):
       #base 2 representation of inputs, to match encodeing in dictionary 
            mapKey = 0
            mapKeyB = 0
            for k in range(degreeList[agent]):     
                mapKey = mapKey + (2**k)*X[t,edgeList[agent][k]]
                mapKeyB = mapKeyB + (2**k)*Xb[t,edgeList[agent][k]]
            X[t+1,agent] = mapList[agent][mapKey]
            Xb[t+1,agent] = mapList[agent][mapKeyB]
        #record stats
        hamDist[t] = HamDist(X[t,],Xb[t,]) / N
        
 #   stateMean = X[t+1,].mean()
   
    return(hamDist)

################################
#%% Iterating ABM over parameters
################################

# Data ####################
output = []
###############################

start_time = time.time()
np.random.seed(97)

edgeList, degreeList = MakeGraph(N,K)

for n in range(N_population):
    for iter in range(nIter):
        x_0 = 0.5 #np.random.uniform(low = 0.3, high = 0.7)
        p = P[n]
        hamDist = kaufNetABM(N,x_0,p,T,edgeList,degreeList)
        #output.append([p_samp,k_samp,stateMean, hamDist])

            
print("---  ABM sims: %s seconds ---" % (time.time() - start_time))


#############################################################
#%% write to csv
output = np.array(output)
output_df = pd.DataFrame(data = output, columns = ['p','k','state_mean','hamming_dist'])


# seed = 97, here

# output_df.to_csv('C:\dev\ABMoutput_examples\kauffman_tseries.csv',index = False)


