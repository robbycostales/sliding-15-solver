# Date: 2018-02-28
# Author: Robby Costales

# Purpose: Module for all things related to heuristics
from misc import *

def heuristicBad(state):
    """
    Placeholder for heuristic function
    """
    return 0


def heuristicNumInPlace(state):
    """
    Heuristic that tells us how many of a given state's tiles are in the right
    the right place. Formerly known as "heuristicMedium"

    Args:
        state :  list of 1d python list representations of S15
    Returns:
        heuristic (int)
    """
    solution = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    return len([i for i in range(16) if state[i] != solution[i]])


def heuristicManhattan(state, goal=None):
    """
    Heuristic that tells us for each tile, how far away it is from it's
    original position. We take the sum of these measurements

    Args:
        state :  list of 1d python list representations of S15
    Returns:
        heuristic (int)
    """

    if goal == None:
        goal =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    else:
        if 0 in goal:
            goal[goal.index(0)] = 16
    # need to make sure one has 16, other has 0
    if 0 in state:
        state[state.index(0)] = 16

    total = 0
    for row in range(4):
        for col in range(4):
            # find index of goal list
            index = 4*row + col
            # find correct value
            correctVal = goal[index]
            # find incorrect value
            incorrectVal = state[index]
            # if 16 we don't care about it
            if incorrectVal == 16: continue
            # figure out row of incorrect
            incorrectIndex = state.index(correctVal)
            incorrectRow = int((incorrectIndex)/4)
            incorrectCol = (incorrectIndex)%4
            distance = abs(incorrectRow-row) + abs(incorrectCol-col)
            total += distance
    return total
