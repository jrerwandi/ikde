import math
import numpy as np
from numba import jit, njit, prange
import warnings
warnings.filterwarnings('ignore')

#@njit
def DE(func, target, angle, link, n_params, lb, ub,Cr= 0.5, F= 0.5, NP=10, max_gen=300):
    
    #inisial populasi secara random
    target_vectors = np.random.uniform(low=lb, high=ub, size=(NP, n_params))
    donor_vector = np.zeros(n_params)
    trial_vector = np.zeros(n_params)
    best_fitness = np.inf
    j_evo = jit(evo, nopython = False)
    best_fitness, angle = j_evo(target_vectors, donor_vector,trial_vector,best_fitness,  func, target, angle, link, n_params, lb, ub)
    '''
    for gen in range(max_gen):
        #print("Generation :", gen)
        for pop in range(NP):
            #mutasi
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(index_choice, 3)
            
            #mutasi    
            donor_vector = target_vectors[a] + F * (target_vectors[b]-target_vectors[c])
            donor_vector = np.clip(donor_vector, lb,ub)
            donor_vector = donor_vector.flatten()
            #print("donor",donor_vector)
            #crossover
            cross_points = np.random.rand(n_params) < Cr            
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            #obj_func
            target_fitness, d = func(target,target_vectors[pop],link)
            trial_fitness, e = func(target,trial_vector,link)
            
            #seleksi
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
                angle = e
            else:
                best_fitness = target_fitness
                angle = d
            best_fitness = np.array(best_fitness)        
            angle = np.array(angle)
    '''
      
    
       
    return best_fitness, angle

NP = 10
F = 0.5 
Cr = 0.5 
max_gen = 300
def evo(target_vectors, donor_vector,trial_vector,best_fitness,  func, target, angle, link, n_params, lb, ub):
           
    for gen in range(max_gen):
        #print("Generation :", gen)
        for pop in range(NP):
            #mutasi
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
            while a == b or a == c or b == c:
                a, b, c = np.random.choice(index_choice, 3)
          
            #mutasi    
            donor_vector = target_vectors[a] + F * (target_vectors[b]-target_vectors[c])
            donor_vector = np.clip(donor_vector, lb,ub)
            donor_vector = donor_vector.flatten()
            #print("donor",donor_vector)
            #crossover
            cross_points = np.random.rand(n_params) < Cr            
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            #obj_func
            target_fitness, d = func(target,target_vectors[pop],link)
            trial_fitness, e = func(target,trial_vector,link)
            
            #seleksi
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
                angle = e
            else:
                best_fitness = target_fitness
                angle = d
            best_fitness = np.array(best_fitness)        
            angle = np.array(angle)
    
        
    
       
    return best_fitness, angle

