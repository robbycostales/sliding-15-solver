# Date: 2018-02-24
# Author: Robby Costales

# Purpose: Creates a dictionary for the walking distance heuristic

import sys
import random
import ast
from queue import PriorityQueue
import time
import copy
import mainAlg as mA


def transpose(og):
    return [list(x) for x in zip(*og)]


def BFS(S, neighborhoodFn):
    """
    Function that performs the graph search. In this case, we want a simple
    breadth first search

    Args:
        S - initial set of states
        neighborhoodFn - returns the neighbors of a given state

    Returns:
        RunTime, Path

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
        explored[str(mA.rankPerm(s))] = 0

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
            boo, rank = mA.rankInExplored(neighbor, explored)
            if str(rank) not in explored:
                # val is the current distance from the goal, so cost increments by 1
                # cost = val + 1
                newPath = path + [neighbor]
                pastCost = len(newPath) - 1
                explored[str(rank)] = pastCost
                frontier.put((pastCost, newPath))

    currentTime = time.time()
    return [currentTime - startTime, None]


def neighbors(state):
    """
    finds VERTICAL neighbors of the state

    returns list of neighbors
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


def createTables():
    global explored
    explored = {}
    # Initial walking distance state
    initial =   [[4, 0, 0, 0],
                [0, 4, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 3]]
    [runTime, path] = BFS([initial], neighbors)

    return explored


if __name__ == "__main__":
    pass
