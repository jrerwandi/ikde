import numba as nb
from time import time, sleep

@nb.njit
def a(j):
    for x in range(30000):
        #t = 1 + 1
#        print(t)
    
        for g in range(1000):
            t =
            #u = 1 + 2
 #           print(u)
    
    return t, h


start = time()   
a()
print(f"finished after {round(time() - start,2)} seconds")
start2 = time()   
a()
print(f"finished after2 {round(time() - start2,2)} seconds")

