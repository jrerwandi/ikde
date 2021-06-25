import numpy as np
from numpy import random ,clip, zeros,inf, where
   
def DE(func, target, angle, link, n_params, lb, ub,Cr=0.9, F=0.9, NP=10, max_gen=200):
    

    #inisial populasi secara random
    target_vectors = random.uniform(low=lb, high=ub, size=(NP, n_params))
    
    donor_vector = zeros(n_params)
    trial_vector = zeros(n_params)
    best_fitness = inf

    for gen in range(max_gen):
        
        #print("Generation :", gen)
        for pop in range(NP):
            #mutasi
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = random.choice(index_choice, 3)
            while a == b or a == c or b == c:
                a, b, c = random.choice(index_choice, 3)
            
            #mutasi    
            donor_vector = target_vectors[a] + F * (target_vectors[b]-target_vectors[c])
            donor_vector = clip(donor_vector, lb,ub)
            donor_vector = donor_vector.flatten()
            #print("donor",donor_vector)
            #crossover
            cross_points = random.rand(n_params) < Cr            
            trial_vector = where(cross_points, donor_vector, target_vectors[pop])
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
        #print("Best fitness :", best_fitness)
        
    
       
    return best_fitness, angle
