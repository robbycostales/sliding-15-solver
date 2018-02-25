import tests as t
import time
import createTables as cT
import random

start = time.time()
for i in range(300000):
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5]
    # x = [0,0,0,1,0,0,2,0,0,0,9,0,4,0,0,8,0,7,0,0]
    rank = cT.rankPerm(x)
cur = time.time()

print(cur-start)
