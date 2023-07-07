# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 14:30:24 2021

Couzin model from Fanqi Zeng, recreating (most of) the dynamics in Couzin (2002)

Changes: 
    a) included a 3rd spatial dimension for the boundary conditions, then promptly 
    comented out boundary condition
    b) created function main_wrapper() to wrap the agent dynamics
    c) wrote group_dir (Fanqi) and group_rho to measure agent responses
    d) added variation in individual agent r_o and r_a.
        agent.r_a , .r_o are now normally distributed with mean r_a, r_o resp
        and standard deviation == 1. r_r ~ N(1,0.1)
    e) paramterized agent speed
    f) add some small (uniform) random noise to agent velocity at each time step
        NOTE: Couzin (2002) uses spherically wrapped gaussian with std = 0.05
        This noise adds quite a bit to convergence times (max_steps > 1000)
    g) double loop over main_wrapper() for r_o_values and r_a_values

I'm helping speed the convergence along by packing the initial positions into the 
center of the field (see class Agent __init__ pos) and removing the boundary conditions
Reflecting BC create edge effects  

Version _a:
    returning time series 
    random initial conditions
    
Version _a1:
     compute space based statistics (gridded)
     writes files
     "C:\dev\ABMoutput_examples\couzin_spacetime_paramkey.csv"
     "C:\dev\ABMoutput_examples\couzin_spacetime_groupdir.csv" 
     "C:\dev\ABMoutput_examples\couzin_spacetime_grouprho.csv"
     
     Here, the statistics (operators) are normalized to the number of agents within
     each grid cell.  So, sum of all grids does not equal non-spatial (or universal)
     statistics
     
Version _a2:
    many interations over parameters: reduce parameter dimension to one
    by tracing along the diagonal, then iterated over each parameter choice
    
    "C:\dev\ABMoutput_examples\couzin_spacetime_paramkey_reps.csv"
    "C:\dev\ABMoutput_examples\couzin_spacetime_groupdir_reps.csv"
    "C:\dev\ABMoutput_examples\couzin_spacetime_grouprho_reps.csv"
     
     
Version _a3:
     compute space based statistics (gridded), in the manner of _a1:
         r_o and r_a roughly linearly spaced between 1 and 14
     writes files
     "C:\dev\ABMoutput_examples\couzin_spacetime_paramkey_conserved.csv"
     "C:\dev\ABMoutput_examples\couzin_spacetime_groupdir3_conserved.csv" 
     "C:\dev\ABMoutput_examples\couzin_spacetime_grouprho3_conserved.csv"
     
     Here, the statistics (operators) are NOT normalized to the number of agents within
     each grid cell, but to the total number of agents.  
     So, sum of all grids (ALMOST) equals non-spatial (or universal)
     statistics (we say almost, because cells with exactly 1 agent are assigned value 0)
     
Version _a5:
    Same as _a3 but with space chopped into 3 x 3 x 3 grid
    
     compute space based statistics (gridded), in the manner of _a1:
         r_o and r_a roughly linearly spaced between 1 and 14
     writes files
     "C:\dev\ABMoutput_examples\couzin_spacetime_paramkey_conserved2.csv"
     "C:\dev\ABMoutput_examples\couzin_spacetime_groupdir3_conserved2.csv" 
     "C:\dev\ABMoutput_examples\couzin_spacetime_grouprho3_conserved2.csv"
     
     Here, the statistics (operators) are NOT normalized to the number of agents within
     each grid cell, but to the total number of agents.  
     So, sum of all grids (ALMOST) equals non-spatial (or universal)
     statistics (we say almost, because cells with exactly 1 agent are assigned value 0)



@author: bruce
"""


import numpy as np
from numpy.linalg import *
from math import *
import time

GRID_SIZE = 11
R_MAX = 14
R_MIN = 1

# INPUT PARAMETERS INTO main_wrapper()
# NOTE: we must have 1 < = r_o < = r_a
dimension = '3d'    # 2d/3d
n = 100             # number of agents
max_steps = 2400
r_o_values = np.linspace(R_MIN,R_MAX,GRID_SIZE)
r_a_values = np.linspace(R_MIN,R_MAX,GRID_SIZE)
# main_wrapper() takes only a single values of r_o and r_a
# r_o_values, r_a_values are all the values we wish to loop over


np.random.seed(65942)

epsilon = np.random.normal(loc=0.0, scale = 0.2, size = [2,GRID_SIZE])

r_o_values = np.round(np.clip(r_o_values + epsilon[0,],0,15),3)
r_a_values = np.round(np.clip(r_a_values + epsilon[1,],0,15),3)


###################################################
#%% class definitions
class Field:
    def __init__(self):
        self.width = 99    # x_max[m]
        self.height = 99   # y_max[m]
        self.depth = 99    # z_max[m]
        
class Agent:
    def __init__(self, agent_id, speed,r_o, r_a, field,dimension):
        self.id = agent_id
        self.pos = np.array([0, 0, 0])
        self.pos[0] = np.random.uniform(field.width*0.25, field.width*0.75)
        self.pos[1] = np.random.uniform(field.height*0.25, field.height*0.75)
        self.pos[2] = np.random.uniform(field.depth*0.25, field.depth*0.75)
        self.vel = np.random.uniform(-1, 1, 3)
        if dimension == '2d':
            self.pos[2] = 0
            self.vel[2] = 0
        self.vel = self.vel / norm(self.vel) * speed
        self.r_r = max([0,np.random.normal(loc = 1, scale = 0.1)])
        self.r_o = max([self.r_r,np.random.normal(loc = r_o,scale = 0.1)])
        self.r_a = max([self.r_o,np.random.normal(loc = r_a,scale = 0.1)])
    def update_position(self, delta_t):
        self.pos = self.pos + self.vel * delta_t
########################################################3
# helper functions
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)
       
def rotation_matrix_about(axis, theta):
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

################################################################
    #
def main_wrapper(n,max_steps,r_o,r_a,dimension):
    
    dt = 0.1
    field_of_view = 3*pi/2
    theta_dot_max = 1
    constant_speed = 3  
    
    swarm = []
    field = Field()
    [swarm.append(Agent(i, constant_speed,r_o,r_a,field,dimension)) for i in range(n)]
    
    t = 0
    
   # group_dir = []
   # group_rho = []
    direction = np.zeros((max_steps,n,3))
    position = np.zeros((max_steps,n,3))
    while t < max_steps:
        
        
        for agent in swarm:
            d = 0
            d_r = 0
            d_o = 0
            d_a = 0
            
            #boundary conditions
            if agent.pos[0] > field.width:
                agent.vel[0] = -agent.vel[0]
            if agent.pos[0] < 0:
                agent.vel[0] = -agent.vel[0]
            if agent.pos[1] > field.height:
                agent.vel[1] = -agent.vel[1]
            if agent.pos[1] < 0:
                agent.vel[1] = -agent.vel[1]
            if agent.pos[2] > field.depth:
                agent.vel[2] = -agent.vel[2]
            if agent.pos[2] < 0:
                agent.vel[2] = -agent.vel[2]
                
            for neighbor in swarm:
                if agent.id != neighbor.id:
                    r = neighbor.pos - agent.pos
                    r_normalized = r/norm(r)
                    norm_r = norm(r)
                    agent_vel_normalized = agent.vel/norm(agent.vel)
                    # print('norm_r', norm_r)
                    if acos(np.dot(r_normalized, agent_vel_normalized)) < field_of_view / 2:
                        if norm_r < agent.r_r:
                            d_r = d_r - r_normalized
                        elif norm_r < agent.r_o:
                            d_o = d_o + neighbor.vel/norm(neighbor.vel)
                        elif norm_r < agent.r_a:
                            d_a = d_a + r_normalized

            if norm(d_r) != 0:
                d = d_r
            elif norm(d_a) != 0 and norm(d_o) != 0:
                d = (d_o + d_a)/2
            elif norm(d_o) != 0:
                d = d_o
            elif norm(d_a) != 0:
                d = d_a
            if norm(d) != 0:
                z = np.cross(d/norm(d), agent.vel/norm(agent.vel))
                angle_between = asin(norm(z))
                if angle_between >= theta_dot_max*dt:
                    rot = rotation_matrix_about(z, theta_dot_max*dt)
                    agent.vel = np.asmatrix(agent.vel) * rot
                    agent.vel = np.asarray(agent.vel)[0]
                elif abs(angle_between)-pi > 0:
                    agent.vel = d/norm(d) * constant_speed
            #add some noise
            agent.vel = agent.vel + np.random.uniform(-0.05,0.05,size = 3)
        [agent.update_position(dt) for agent in swarm]
           
        #t = t+1
        for agent in range(n):
           direction[t,agent,:] = swarm[agent].vel
           position[t,agent,:] = swarm[agent].pos
        
        
        t = t+1
    # THIN HERE
    direction = direction[0:max_steps:24,:,:]
    position = position[0:max_steps:24,:,:]
    
    thin_steps = position.shape[0]
    #Partition into 4 cells (x 3 dimensions), ASSUME Field.width == 100
    
    #TODO
    # make partitions into 3 x 3 x 3
    
    local_group_rho = np.zeros((27,thin_steps))
    local_group_dir = np.zeros((27,thin_steps))
    #centroid = np.zeros((n,3))
    
    partition = np.floor(position) // 33
    
    
    
    for tt in range(thin_steps):
        partition_index = 0
        for x_grid in range(3):
            for y_grid in range(3):
                for z_grid in range(3):
                    logic_partition = np.logical_and(partition[tt,:,0]==x_grid,partition[tt,:,1]==y_grid)
                    logic_partition = np.logical_and(logic_partition,partition[tt,:,2] == z_grid)
                    
                    sum_vel0, sum_vel1, sum_vel2 = 0, 0, 0
                    group_center = 0
                    
                    local_group_size = 0
                    for agent in range(n):
                        if logic_partition[agent]:
                            sum_vel0 += direction[tt,agent,0]/((direction[tt,agent,0]**2 + direction[tt,agent,1]**2+direction[tt,agent,2]**2)**0.5)
                            sum_vel1 += direction[tt,agent,1]/((direction[tt,agent,0]**2 + direction[tt,agent,1]**2+direction[tt,agent,2]**2)**0.5)
                            sum_vel2 += direction[tt,agent,2]/((direction[tt,agent,0]**2 + direction[tt,agent,1]**2+direction[tt,agent,2]**2)**0.5)
                            group_center += position[tt,agent,]
                            local_group_size += 1
                    if local_group_size > 1:    
                        group_center = group_center / local_group_size
                        sum_rho = 0
                        for agent in range(n):
                            if logic_partition[agent]:
                                dir_to_center = position[tt,agent,] - group_center
                                dir_to_center = unit_vector(dir_to_center)
                                agent_velocity = unit_vector(direction[tt,agent,])
                                agent_rho = np.cross(agent_velocity,dir_to_center)
                                sum_rho += agent_rho
                    # else: local_group_rho[partition_index,tt] = 0, local_group_dir
                        # dividing by all agents, so that sum over partitions is (almost) conserved
                        local_group_dir[partition_index,tt] = (sum_vel0**2 + sum_vel1**2+ sum_vel2**2)**0.5 / n
                        local_group_rho[partition_index,tt] = np.linalg.norm(sum_rho) / n
                    partition_index += 1   
                    
                    
    return([local_group_dir,local_group_rho])

#################################
#%% looping over parameters



#start_time = time.time()

code_params = np.zeros((27*GRID_SIZE**2,5))
# 64 spatial compartments, 121 parameter values
dir_output = np.zeros([27*GRID_SIZE**2,100])
rho_output = np.zeros([27*GRID_SIZE**2,100])
   
centroid = np.zeros((27,3))
part_counter = 0
for x_grid in range(3):
    for y_grid in range(3):
        for z_grid in range(3):
            centroid[part_counter,:] =  16.5+33*np.asarray([x_grid,y_grid,z_grid])
            part_counter += 1
            
            
start_time = time.time() 
counter = 0
for r_o in r_o_values:
    for r_a in r_a_values:
        # r_o = np.round(r_o+np.random.uniform(low = -0.1,high = 0.1),3)
        # r_a = np.round(r_a+np.random.uniform(low = -0.1,high = 0.1),3)
        r_a_input = r_o + r_a 
        #print(r_o,r_a)
        abm_out = main_wrapper(n,max_steps,r_o,r_a_input,dimension)
        
        code_params[27*counter:27*(counter+1),:] = np.transpose(np.asarray([np.transpose(np.tile(r_o,27)),np.transpose(np.tile(r_a,27)),centroid[:27,0],centroid[:27,1],centroid[:27,2]]))
        dir_output[27*counter:27*(counter+1),] = abm_out[0]
        rho_output[27*counter:27*(counter+1),] = abm_out[1]
        counter += 1
        print(counter)
        
print("---Run Time: %s seconds ---" % (time.time() - start_time)) 


#%%
import pandas as pd

param_array = np.asarray(code_params)
param_df = pd.DataFrame(param_array,columns = ["r_o","r_a","x","y","z"])
dir_df = pd.DataFrame(dir_output)
rho_df = pd.DataFrame(rho_output)

param_df.to_csv("C:\dev\ABMoutput_examples\couzin_spacetime_paramkey_conserved2.csv")
dir_df.to_csv("C:\dev\ABMoutput_examples\couzin_spacetime_groupdir_conserved2.csv") 
rho_df.to_csv("C:\dev\ABMoutput_examples\couzin_spacetime_grouprho_conserved2.csv") 
 
#%%

"""
import matplotlib.pyplot as plt

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


tsteps = np.arange(max_steps)
window = 100
# r_o_values = [2,5,15]
# r_a_values = [10,15,25]

ma0 = movingaverage(X[0,],window)
ma1 = movingaverage(X[1,],window)
ma2 = movingaverage(X[2,],window)

fig, ax = plt.subplots()
plt.plot(tsteps,X[0,:pMax],alpha = 0.8,label = '$r_o = 2, r_a = 8$')
plt.plot(tsteps[window:pMax],ma0[window:pMax],color = 'black')
plt.plot(X[1,:pMax],alpha = 0.8,label = '$r_o = 5, r_a = 15$')
plt.plot(tsteps[window:pMax],ma1[window:pMax],color = 'black')
plt.plot(X[2,:pMax],alpha = 0.8,label = '$r_o = 15, r_a = 25$')
plt.plot(tsteps[window:pMax],ma2[window:pMax],color = 'black')
plt.xlabel('Time')
plt.ylabel('Group direction')
plt.ylim([0,1])
plt.xlim([0,pMax+200])
ax.legend()
"""