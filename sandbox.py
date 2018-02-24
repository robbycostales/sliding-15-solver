import tests as t
import time
import createTables as cT

initial =   [[14, 4, 11, 10],
            [12, 9, 13, 5],
            [1, 2, 7, 15],
            [3, 8, 16, 6]]

state = [14, 4, 11, 10, 12, 9, 13, 5, 1, 2, 7, 15, 3, 8, 16, 6]

r1 = cT.rankPerm(initial)
r2 = cT.rankPerm(state)

print(r1)
print(r2)
