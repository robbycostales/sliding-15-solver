# Date: 2018-02-24
# Author: Robby Costales

# Purpose: TESTING
import time
import random

# local
from run import *
from createWD import *
from heuristics import *
from misc import *

flat = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

flat2 = [1,2,4,3,6,5,7,8,9,10,11,12,13,14,15,16]

x = []
for i in range(100):
    state = copy.deepcopy(flat)
    random.shuffle(state)
    while not isSolvable(state):
        random.shuffle(state)
    x.append(state)

for i in range(len(x)):
    print(str(x[i]) + ",")
