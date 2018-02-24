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
        explored[str(rankPerm(s))] = 0

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
            boo, rank = rankInExplored(neighbor, explored)
            if str(rank) not in explored:
                # val is the current distance from the goal, so cost increments by 1
                cost = val + 1
                explored[str(rank)] = cost
                newPath = path + [neighbor]
                frontier.put((cost, newPath))

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


def rankInExplored(state, dictionary):
    """
    Checks if a state's rank is in given dictionary
    """
    rank = rankPerm(state)
    if str(rank) in dictionary:
        return True, rank
    else:
        return False, rank


def rankPerm(perm, inverse = None, m = None):
    """
    rankPerm(perm) returns the rank of permutation perm.
    The rank is done according to Myrvold, Ruskey "Ranking and unranking permutations in linear-time".
    perm should be a 1-based list, such as [1,2,3,4,5].

    However, this function will automatically flatten a 2d array into a 1-based list
    """

    # Robby's Edits:
    if type(perm[0]) == type([]):
        # flattens 2d array
        perm = sum(perm, [])
    # end of Robby's edits


    # if the parameters are None, then this is the initial call, so set the values
    if inverse == None:
        perm = list(perm) # make a copy of the perm; this algorithm will sort it
        m = len(perm)
        inverse = [-1]*m
        for i in range(m):
            inverse[perm[i]-1] = i+1

    if m == 1:
        return 0
    s = perm[m-1]-1
    x = m-1
    y = inverse[m-1]-1
    temp = perm[x]
    perm[x] = perm[y];
    perm[y] = temp;
    x = s
    y = m-1
    temp = inverse[x]
    inverse[x] = inverse[y]
    inverse[y] = temp
    return s + m * rankPerm(perm, inverse, m-1)


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
