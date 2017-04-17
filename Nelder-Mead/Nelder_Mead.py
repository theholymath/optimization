## Ryan Croke
## 4/17/2017

# Original Nelder-Mead Paper: http://www.ime.unicamp.br/~sandra/MT853/handouts/Ref3(NelderMead1965).pdf
# Implemented from: http://www.webpages.uidaho.edu/~fuchang/res/ANMS.pdf
# and borrowed a few things from scipy ;)

import math
import copy
import numpy as np
import pandas as pd

def Nelder_Mead(x0,func,
                step_size = 0.1,max_iterations=470,
                tol = 10e-6,no_improve_break=20):
    
    alpha = 1.0 # reflection
    beta  = 2.0 # contraction
    gamma = 0.5 # expansion
    
    N       = len(x0)
    x0      = np.asfarray(x0) # make sure it's float
    best    = func(x0)
    simplex = np.asfarray(np.append(x0,best))

    # for adaptive NM use
    # only use for large N
    if N > 15:
        alpha = 1.0
        beta  = 1.0 + 2.0/float(N)
        gamma = 0.75 - 1.0/(2.0*float(N))
        delta = 1.0 - 1.0/float(N)
    
    # build initial simplex
    for i in range(int(N)):
        x = np.array(x0, copy=True)
        if x[i] != 0:
            x[i] = (1 + 0.05)*x[i] # from scipy
        else:
            x[i] = 0.00025
        simplex = np.concatenate((simplex,np.append(x,func(x))))
    simplex = simplex.reshape(int(len(simplex)/(N+1)),(N+1))
    simplex = simplex[simplex[:,N].argsort()] # ascending


    no_improve = 0
    iterations = 0
    while True:
        # sort vertices by function value
        simplex = simplex[simplex[:,N].argsort()] # ascending
        min_value = simplex[0,-1]
        
        if iterations >= max_iterations and max_iterations >= max_iterations:
            print("hit max iterations")
            return simplex[0,:]
            
        iterations += 1
        
        # break after no_improv_break iterations with no improvement
        #print('...best so far:', min_value)

        if min_value < best - tol:
            no_improve = 0
            best = min_value
        else:
            no_improve += 1

        if no_improve >= no_improve_break:
            print("Converged")
            return simplex[0,:]
        
        centroid = np.array([np.sum(simplex[:-1,i])/N for i in range(N)])
        
        # relection
        x_reflection = (1 + alpha)*centroid - alpha*simplex[-1,:-1]
        y_reflection = func(x_reflection)
        execute_shrink = 0
        
        # expansion
        if y_reflection < simplex[0,-1]:
            x_expansion = (1 + beta)*centroid - beta*simplex[-1,:-1]
            y_expansion = func(x_expansion)

            if y_expansion < y_reflection: # mistake in NL original paper at this step
                simplex[-1,:] = np.append(x_expansion,y_expansion)
                continue
            else:    
                simplex[-1,:] = np.append(x_reflection,y_reflection)
                continue
        
        else: # refelection is bigger than current min
            if  y_reflection < simplex[-2,-1]: # different than original paper
                simplex[-1,:] = np.append(x_reflection,y_reflection)
                continue
            else: # reflection is bigger than simplex[-2,-1]
                # contraction
                if y_reflection < simplex[-1,-1]:
                    x_contraction_o = -gamma*simplex[-1,:-1] + (1 + gamma)*centroid
                    y_contraction_o = func(x_contraction_o)
                    
                    if y_contraction_o <= y_reflection:
                        simplex[-1,:] = np.append(x_contraction_o,y_contraction_o)
                        continue
                    else:
                        execute_shrink = 1
                else:
                    x_contraction_i = gamma*simplex[-1,:-1] + (1 - gamma)*centroid
                    y_contraction_i = func(x_contraction_i)
                    
                    if y_contraction_i < simplex[-1,-1]:
                        simplex[-1,:] = np.append(x_contraction_i,y_contraction_i)
                        continue
                    else:
                        execute_shrink = 1
                    
                    if execute_shrink:
                        # shrink
                        low_vertex  = simplex[0,:-1]
                        new_simplex = np.zeros((N+1,N+1))
                        
                        for i,row in enumerate(simplex[:,:-1]):
                            x_new = low_vertex + (row - low_vertex)/2.0
                            y_new = func(x_new)
                            new_simplex[i,:] = np.asarray(np.append(x_new,y_new)) # maybe overwriting in loop....
                        simplex = new_simplex
                        simplex = simplex[simplex[:,N].argsort()]
                        continue
