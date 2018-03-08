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


def convertWD(state, goal = None, orientation="vert"):
    """
    Given S15 state, converts to WD state (default is for vertical)

    Args:
        state : 1-d python representation of S15 state
        goal : 1-d rep of goal state
        orientation : vertical (vert) vs horizontal (horiz) walking distance
    Returns:
        rank

        rank is rank of WD state (to be searched in table created by
        createTables), and state is just the converted state in matrix form.

    """

    # zero because we want 4, 4, 4, 3 in goal rep not 4, 4, 4, 4
    # would normally be 16
    if goal == None:
        goal =  [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]]
    else:
        # MUST REPLACE 16 WITH 0 SO WE GET THE NEEDED 4, 4, 4, 3 or other permuation
        # otherwise we get 4, 4, 4, 4
        if 16 in goal:
            goal[goal.index(16)] = 0

        goal = unFlatten(goal)

    # need to make sure one has 16, other has 0
    if 0 in state:
        state[state.index(0)] = 16

    # converting 1-list to 2-list
    conv = unFlatten(state)

    if orientation == "horiz":
        goal = transpose(goal)
        conv = transpose(conv)

    # # check intersection in each row, create 1-D list to rank
    # ints = []
    # for i in conv:
    #     for j in goal:
    #             ints.append(len(set(i).intersection(j)))
    #
    #
    # # find rank of the WD state created
    # rank = rankPerm(ints)
    # return rank


    # JR'S STUFF
    for i in range(16):
        doNothing([])


    matrix = [0] * 4
    for i in range(4):
        matrix[i] = [0] * 4

    for row in range(4):
        for col in range(4):
            index = 4 * row + col
            value = state[index]
            if value == 16:
                continue
            elif value in goal[0]:
                matrix[row][0] += 1
            elif value in goal[1]:
                matrix[row][1] += 1
            elif value in goal[2]:
                matrix[row][2] += 1
            elif value in goal[3]:
                matrix[row][3] += 1
    rank = rankPerm(matrix)
    return rank
