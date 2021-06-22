import numpy as np
NP = 5

lb = np.array([(-60, -10, 0 , -180)])
ub = np.array([(180, 90, 160 , 180)])
n_params = 4

target_vectors = np.random.uniform(low=lb, high=ub, size=(NP, n_params))
target_vectors = target_vectors * target_vectors
b = np.clip(target_vectors,lb,ub)
print(b)
