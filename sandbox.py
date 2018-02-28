# Date: 2018-02-24
# Author: Robby Costales

# Purpose: TESTING
import time
import random
import numpy as np

# local
import createTables as ct
import mainAlgs as ma
import funcs as fu

flat = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
state =  [[1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]]

r1 = [1, 2, 3, 4]

r2 = [4, 6, 3, 2]

npFlat = np.array(flat)
npState = np.array(state)
npr1 = np.array(r1)
npr2 = np.array(r2)


print("to string")

start = time.time()

for i in range(2000000):
    x = str(state)
    y = str(flat)


end = time.time()
print(str(end-start), "\n")



print("freeze")

start = time.time()

for i in range(2000000):
    x = tuple(state)
    y = tuple(flat)

end = time.time()
print(str(end-start), "\n")
