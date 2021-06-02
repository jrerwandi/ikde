import math
import numpy as np
    
    
def DE(func, target, angle, link, n_params, lb, ub,Cr=0.5, F=0.5, NP=20, max_gen=300):
    
#    lb = [(-np.radians(60), -np.pi/2, 0 , -np.pi)]
#    ub = [(np.pi, np.radians(45), (np.radians(160)) , np.pi)]
    
    target_vectors = np.random.uniform(low=lb, high=ub, size=(NP, n_params))
    
    donor_vector = np.zeros(n_params)
    trial_vector = np.zeros(n_params)
    
    best_fitness = np.inf
    list_best_fitness = []
    for gen in range(max_gen):
        print("Generation :", gen)
        for pop in range(NP):
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
         
                
            donor_vector = target_vectors[a] + F * (target_vectors[b]-target_vectors[c])

            cross_points = np.random.rand(n_params) < Cr
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            
            target_fitness, d = func(target,target_vectors[pop],link)
            trial_fitness, e = func(target,trial_vector,link)
            
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
                angle = d
            else:
                best_fitness = target_fitness
                angle = e
        print("Best fitness :", best_fitness)
        
    
       
    return best_fitness, angle

  
    
    

