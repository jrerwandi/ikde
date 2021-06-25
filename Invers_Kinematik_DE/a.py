import numba as nb
from time import time, sleep

#@nb.njit
def a():
    for x in range(300):
        t = 1 + 1
     #   print(t)
    
        for g in range(1000):
            u = 1 + 2
      #      print(u)
    
    return t


start = time()   
a()
print(f"finished after {round(time() - start,2)} seconds")

