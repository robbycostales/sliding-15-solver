# Date: 2018-02-24
# Author: Robby Costales

# Purpose: Creates a dictionary for the walking distance heuristic

import sys
import random
import ast
from queue import PriorityQueue
import time
import copy
import pickle
# local
import funcs as fu



def BFS(S, neighborhoodFn):
    """
    Function that performs the graph search. In this case, we want a simple
    breadth first search

    Args:
        S (nested list) - initial set of states
        neighborhoodFn - returns the neighbors of a given state

    Returns:
        (runTime, Path) where runTime is the length of the search in seconds
        and path is a list of states from the initial state to the goal state


        if error: -1, None
    """

    # start time of the search
    startTime = time.time()

    # queue for the frontier
    frontier = PriorityQueue()

    for s in S:
        # put the starting state in the frontier
        frontier.put((0, [s]))
        # set to 0 because the starting state is the goal state
        explored[fu.rankPerm(s)] = 0

    while frontier.qsize() > 0:
        # while there is stuff in the frontier
        # grab a priority item from the queue
        (val, path) = frontier.get()
        # node is the last node in the path we pull out
        node = path[-1]
        # we don't have a goal node, so no reason to check for one

        neighborhood = neighborhoodFn(node)
        # we get list of neighbors of our state
        for neighbor in neighborhood:
            boo, rank = fu.stateInDict(neighbor, explored)
            if rank not in explored:
                # val is the current distance from the goal, so cost increments by 1
                # cost = val + 1
                newPath = path + [neighbor]
                pastCost = len(newPath) - 1
                explored[rank] = pastCost
                frontier.put((pastCost, newPath))

    currentTime = time.time()
    return [currentTime - startTime, None]


def neighbors(state):
    """
    Finds VERTICAL neighbors of the state (can submit transpose of the state
    into this function, then transpose result to find horizontal neighbors)

    Args:
        WD state

    Returns:
        list of neighbors
    """
    # list of new neighbors to be returned after function completion
    neighborhood = []

    ### find which row blank is in
    for i in range(4):
        if sum(state[i]) == 3:
            # if the total is 3 instead of 4, our blank is in the ith row
            blankr = i

    ### move blank up: 4 possibilities
    # if blank in top row:
    if blankr==0:
        # cannot move the blank up
        pass
    else:
        # for column in the above row
        for i in range(4):
            # try to move a value down to blankr
            # check if nonzero value in current position above
            if state[blankr-1][i] != 0:
                # create new copy of state
                newState = copy.deepcopy(state)
                # increase blank row at position i
                newState[blankr][i] += 1
                # decrease row above blank row at position i
                newState[blankr-1][i] -= 1
                # add new state to neighborhood
                neighborhood.append(newState)

    ### move blank down: 4 possibilities
    # if blank in bottom row:
    if blankr==3:
        # cannot move the blank down
        pass
    else:
        # for column in the below row
        for i in range(4):
            # try to move a value up to blankr
            # check if nonzero value in current position below
            if state[blankr+1][i] != 0:
                # create new copy of state
                newState = copy.deepcopy(state)
                # increase blank row at position i
                newState[blankr][i] += 1
                # decrease row below blank row at position i
                newState[blankr+1][i] -= 1
                # add new state to neighborhood
                neighborhood.append(newState)

    return neighborhood


def createTables(typ=4):
    """
    Function that uses the above functions to generate the vertical walking
    distance dictionary. Technically function should be called "create dictionary"
    but createTables is more distinct, as there are many dictionaries in use

    Args:
        typ : type of table to be created. 4 is for the og goal state, but using
            other states as goals messes this up. The 16 can be in any column, so
            the '3' is not necessarily in the last diagonal

    Returns:
        explored : the dictionary where keys are ranks of the WD matrices (for
        vertical WD, and the values are how many steps from the goal state)
    """
    global explored
    explored = {}
    # Initial walking distance state
    print(typ)
    if typ==1:
        initial =   [[3, 0, 0, 0],
                    [0, 4, 0, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 4]]
    if typ==2:
        initial =   [[4, 0, 0, 0],
                    [0, 3, 0, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 4]]
    if typ==3:
        initial =   [[4, 0, 0, 0],
                    [0, 4, 0, 0],
                    [0, 0, 3, 0],
                    [0, 0, 0, 4]]
    if typ==4:
        initial =   [[4, 0, 0, 0],
                    [0, 4, 0, 0],
                    [0, 0, 4, 0],
                    [0, 0, 0, 3]]

    [runTime, path] = BFS([initial], neighbors)

    return explored


if __name__ == "__main__":
    print("creating tables...")
    vertWDRanks1 = createTables(typ=1)
    vertWDRanks2 = createTables(typ=2)
    vertWDRanks3 = createTables(typ=3)
    vertWDRanks4 = createTables(typ=4)
    print(str(len(vertWDRanks1)) + " items created in table 1")
    print(str(len(vertWDRanks1)) + " items created in table 2")
    print(str(len(vertWDRanks1)) + " items created in table 3")
    print(str(len(vertWDRanks1)) + " items created in table 4\n")


    TABLES = [vertWDRanks1, vertWDRanks2, vertWDRanks3, vertWDRanks4]

    pickle_out = open("TABLES","wb")
    pickle.dump(TABLES, pickle_out)
    pickle_out.close()
    pass
