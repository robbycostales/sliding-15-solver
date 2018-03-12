# Date: 2018-02-25
# Author: Robby Costales

# Purpose: Contains all of the main solving algorithms

import sys
import random
import time
import copy
import pickle
from statistics import median
import matplotlib.pyplot as plt
# local
from createWD import *
from misc import *
from heuristics import *
# profiling
import cProfile
# queue stuff
from queue import PriorityQueue
import heapq as hq



def neighbors(state, vertMat, horizMat, goal):
    """
    Finds neighbors of an S15 state

    Args:
        state : 1d python list representation of S15
    Returns:
        list : list of neighbors (where each neighbor in given state format)
    """
    neighborhood = []

    # find blank position

    try:
        i = state.index(16)
    except:
        i = state.index(0)
        state[i] = 16

    # move blank left? WIP
    if i % 4 != 0:
        # print("Left")
        # only change horiz
        # horizMatMod = copy.deepcopy(horizMat)
        horizMatMod = horizMat[:]
        # find index of piece replaced by 16 (currently left of 16) in goal
        iC = goal.index(state[i-1])
        # find column of of piece replaced by 16 ( in goal)
        cC = iC % 4
        # find column of 16 in state
        c16 = i % 4
        # since 16 is moved left, we add to lower
        horizMatMod[cC+c16*4] += 1
        # since 16 is moved left, we subtract from upper
        horizMatMod[cC+(c16-1)*4] -= 1

        newState = state[:i-1] + [state[i],state[i-1]] + state[i+1:]
        neighborhood.append((newState, vertMat, horizMatMod))

    # move blank right? WIP
    if i % 4 != 3:
        # print("Right")
        # only change horiz
        # horizMatMod = copy.deepcopy(horizMat)
        horizMatMod = horizMat[:]
        # find index of piece replaced by 16 (currently left of 16) in goal
        iC = goal.index(state[i+1])
        # find column of of piece replaced by 16 ( in goal)
        cC = iC % 4
        # find column of 16 in state
        c16 = i % 4
        # since 16 is moved left, we add to lower
        horizMatMod[cC+c16*4] += 1
        # since 16 is moved left, we subtract from upper
        horizMatMod[cC+(c16+1)*4] -= 1

        newState = state[:i] + [state[i+1],state[i]] + state[i+2:]
        neighborhood.append((newState, vertMat, horizMatMod))

    # move blank up? WIP
    if i > 3:
        # print("Up")
        # only change vert
        # vertMatMod = copy.deepcopy(vertMat)
        vertMatMod = vertMat[:]
        # find index of piece replaced by 16 (currently above 16) in goal
        iC = goal.index(state[i - 4])
        # find row of of piece replaced by 16 ( in goal)
        rC = iC // 4
        # find row of 16 in state
        r16 = i // 4
        # since 16 is moved up, we add to lower
        vertMatMod[rC+(r16-1)*4] -= 1
        # since 16 is moved up, we subtract from upper
        vertMatMod[rC+r16*4] += 1

        newState = state[:i-4] + [state[i]] + state[i-3:i] + [state[i-4]] + state[i+1:]
        neighborhood.append((newState, vertMatMod, horizMat))

    # move blank down? WIP
    if i < 12:
        # print("Down")
        # only change vert
        # vertMatMod = copy.deepcopy(vertMat)
        vertMatMod = vertMat[:]
        # find index of piece replaced by 16 (currently below 16) in goal
        iC = goal.index(state[i + 4])
        # find row of of piece replaced by 16 (in goal)
        rC = iC // 4
        # find row of 16 in state
        r16 = i // 4
        # since 16 is moved down, we subtract from lower
        vertMatMod[rC+(r16+1)*4] -= 1
        # since 16 is moved down, we add to upper
        vertMatMod[rC+(r16)*4] += 1

        newState = state[:i] + [state[i+4]] + state[i+1:i+4] + [state[i]] + state[i+5:]
        neighborhood.append((newState, vertMatMod, horizMat))

    return neighborhood


def print15(state):
    """
    Pretty prints given S15 state

    Args:
        state : 1d python list representation of S15
    """
    for row in range(4):
        for col in range(4):
            if state[4*row+col] < 10:
                    sys.stdout.write(" ")
            sys.stdout.write(str(state[4*row+col]))
            sys.stdout.write("\t")
        print("")
    return


def print15s(path):
    """
    Pretty prints path (or list) of S15 states

    Args:
        state : list of 1d python list representations of S15
    """
    for i, state in enumerate(path):
        print("step " + str(i))
        print15(state)
        print("")


# this function would be in misc except for explicit use of neighbors function
def scrambler(state, n):
    """
    Scrambles an S15 puzzle state (easier to solve than random)

    Args:
        state :  list of 1d python list representations of S15
        n : number of scrambles (actual distance away will vary)
    Returns:
        scrambled state
    """
    sExplored = {}
    try:
        for step in range(n):
            neighborList = neighbors(state)

            # # don't regenerate previously generated states
            newNeighbors = []
            for i in neighborList:
                boo, rank = stateInDict(i, sExplored)
                if not boo:
                    newNeighbors.append(i)
            neighborList = newNeighbors

            num = len(neighborList)
            nextNeighbor = neighborList[random.randint(0, num-1)]
            state = nextNeighbor

            boo, rank = stateInDict(state, sExplored)
            sExplored[rank] = 1
    except:
        scrambler(state, n)
    return state


def AStar(S, neighborhoodFn, goalFn, visitFn, heuristicFn):
    """
    Searches for goal node for S15 given initial set of states
    Uses AStar search

    Args:
        S : list of starting state(s)
        neighborhoodFn : function that finds neighbors of a given state
        goalFn : tests if a state is a goal or not
        visitFn : visits a given path
        heuristicFn : heuristic for AStar search
    Returns:
        runTime, path

        If something goes wrong, returns: -1, None
    """
    global maxTime
    startTime = time.time()

    frontier = PriorityQueue()

    for s in S:
        frontier.put((0, [s]))
        explored[rankPerm(s)] = 1

    while frontier.qsize() > 0:
        (_, path) = frontier.get()
        node = path[-1]

        # check time
        currentTime = time.time()
        if currentTime - startTime > maxTime:
            return [-1, None]

        if goalFn(node):
            visitFn(path)
            currentTime = time.time()
            return [currentTime - startTime, path]
        else:
            neighborhood = neighborhoodFn(node)
            for neighbor in neighborhood:
                boo, rank = stateInDict(neighbor, explored)
                if not boo:
                    explored[rank] = 1
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    futureCost = heuristicFn(neighbor)
                    totalCost = pastCost + futureCost
                    frontier.put((totalCost, newPath))

    return [-1, None]


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
    ints = []
    for i in conv:
        for j in goal:
                ints.append(len(set(i).intersection(j)))


    # find rank of the WD state created
    rank = rankPerm(ints)
    return rank, ints


# function here because needs dictionaries
def chooseWDDict(state, ori="vert"):
    """
    Given a state, which walking distance type should be used?

    The og goal state has diagonals 4, 4, 4, 3
    However, possible starting states can have all other permuations.

    Type 1: 3, 4, 4, 4
    Type 2: 4, 3, 4, 4
    Type 3: 4, 4, 3, 4
    Type 4: 4, 4, 4, 3

    Args:
        state: S15 puzzle in 1-d python list format
    Returns:
        the dictionary that should be used for the given type
    """
    state = unFlatten(state)
    if ori == "horiz":
        state = transpose(state)
    # no need to transpose, vertical MEANS ROWS!!!! not columns
    # this is confusing af
    # state = fu.transpose(state)
    for i in range(4):
        if 16 in state[i] or 0 in state[i]:
            typ = i+1
            break

    if typ == 1:
        return vertWDRanks1
    elif typ == 2:
        return vertWDRanks2
    elif typ == 3:
        return vertWDRanks3
    elif typ == 4:
        return vertWDRanks4
    else:
        return -1


# function here because we need the dictionaries
def heuristicWD(state, goal=None, typ=4):
    """
    Given a state, what is the walking distance heuristic value?

    Args:
        state : 1-d python representation of S15 state
        goal (optional): if typical goal, let it equal to None
    Returns:
        heuristic (int)
    """

    if goal==None:
        goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    vertWDRanks = chooseWDDict(goal, ori="vert")
    vertRank, vertMat = convertWD(state, goal, "vert")
    x = vertWDRanks[vertRank]

    vertWDRanks = chooseWDDict(goal, ori="horiz")
    horizRank, horizMat = convertWD(state, goal, "horiz")
    y = vertWDRanks[horizRank]

    return x+y, vertMat, horizMat


def neighborsWD(vertMat, horizMat, goal=None):
    if goal==None:
        goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    vertWDRanks = chooseWDDict(goal, ori="vert")
    x = vertWDRanks[rankPerm(vertMat)]

    vertWDRanks = chooseWDDict(goal, ori="horiz")
    y = vertWDRanks[rankPerm(horizMat)]
    return x+y


def bidirectional(S, G, neighborhoodFn, goalFn, visitFn, heuristicFn):
    """
    Searches for goal node for S15 given initial set of states
    Uses bidirectional search

    Args:
        S : list of starting state(s)
        G : list of goal state(s)
        neighborhoodFn : function that finds neighbors of a given state
        goalFn : tests if a state is a goal or not
        visitFn : visits a given path
        heuristicFn : heuristic--should be able to use custom goal state
    Returns:
        runTime, path

        If something goes wrong, returns: -1, None
    """
    global maxTime
    global pastCostConst
    global exploredTo
    global exploredFrom
    global explored

    exploredTo = {}
    explored = {}
    exploredFrom = {}

    startTime = time.time()

    fConst = pastCostConst
    bBound = BRANCH_BOUND

    # # using PQ
    # # paths to goal state
    # frontierTo = PriorityQueue()
    #
    # # paths from goal state
    # frontierFrom = PriorityQueue()

    frontierTo = []

    frontierFrom = []

    # put initial state's path in TO queue
    for s in S:
        # frontierTo.put((0, [s]))
        rank, vert, horiz = heuristicWD(s, goal=G[0])
        hq.heappush(frontierTo, (0, ([s], vert, horiz)))
        exploredTo[rankPerm(s)] = 1

    # put initial state's path in TO queue
    for g in G:
        # frontierFrom.put((0, [g]))
        rank, vert, horiz = heuristicWD(g, goal=S[0])
        hq.heappush(frontierFrom, (0, ([g], vert, horiz)))
        exploredFrom[rankPerm(g)] = 1

    count = 0
    # while frontierFrom.qsize() > 0 and frontierTo.qsize() > 0:
    while len(frontierFrom) > 0 and len(frontierTo) > 0:
        # print(frontierTo.qsize(), frontierFrom.qsize(), end="\r")
        count += 1
        # check time
        currentTime = time.time()
        if currentTime - startTime > maxTime:
            return [-1, None]

        ### TO ------->>>>>
        # pull from TO queue
        # (val, path) = frontierTo.get()
        (val, (path, vertMat, horizMat)) = hq.heappop(frontierTo)


        node = path[-1]

        # check if node in other dictionary
        boo, rank = stateInDict(node, exploredFrom)
        if boo:
            currentTime = time.time()
            # have our final pathTo
            pathTo = path
            # find the pathFrom that has this overlapping node
            # while frontierFrom.qsize() > 0:
            while len(frontierFrom) > 0:
                # (_, pathCheck) = frontierFrom.get()
                (_, (pathCheck, _, _)) = hq.heappop(frontierFrom)
                for i in range(len(pathCheck)):
                    if rank == rankPerm(pathCheck[i]):
                        # we found our pathFrom
                        pathFrom = pathCheck
                        # need to merge these paths on i (i cant appear in both)
                        finalPath = pathTo + pathFrom[i+1::-1]
                        visitFn(finalPath)

                        return [currentTime - startTime, finalPath]
        else:
            neighborhood = neighborhoodFn(node, vertMat, horizMat, G[0])
            for tup in neighborhood:
                (neighbor, vert, horiz) = tup
                boo, rank = stateInDict(neighbor, exploredTo)
                if not boo:
                    exploredTo[rank] = 1
                    newPath = path + [neighbor]
                    pastCost = (len(newPath)-1)//fConst
                    futureCost = neighborsWD(vert, horiz, goal=None)
                    totalCost = pastCost + futureCost
                    if len(newPath) < bBound:
                        # frontierTo.put((totalCost, newPath))
                        hq.heappush(frontierTo, (totalCost, (newPath, vert, horiz)))

        ### <<<<<------- FROM
        # pull from FROM queue
        if count % fromConst == 0:
            # (val, path) = frontierFrom.get()
            (val, (path, vertMat, horizMat)) = hq.heappop(frontierFrom)
            node = path[-1]

            # check if node in other dictionary
            boo, rank = stateInDict(node, exploredTo)
            if boo:
                currentTime = time.time()
                # have our final pathFrom
                pathFrom = path
                # find the pathTo that has this overlapping node
                # while frontierTo.qsize() > 0:
                while len(frontierTo) > 0:
                    # (_, pathCheck) = frontierTo.get()
                    (_, (pathCheck, _, _)) = hq.heappop(frontierTo)
                    for i in range(len(pathCheck)):
                        if rank == rankPerm(pathCheck[i]):
                            # we found our pathTo
                            pathTo = pathCheck
                            # need to merge these paths on i (i cant appear in both)
                            finalPath = pathTo[:i] + pathFrom[::-1]
                            visitFn(finalPath)

                            return [currentTime - startTime, finalPath]
            else:
                neighborhood = neighborhoodFn(node, vertMat, horizMat, S[0])
                for tup in neighborhood:
                    (neighbor, vert, horiz) = tup
                    boo, rank = stateInDict(neighbor, exploredFrom)
                    if not boo:
                        exploredFrom[rank] = 1
                        newPath = path + [neighbor]
                        pastCost = (len(newPath)-1)//fConst
                        futureCost = neighborsWD(vert, horiz, goal=S[0])
                        totalCost = pastCost + futureCost
                        if len(newPath) < bBound:
                            hq.heappush(frontierFrom, (totalCost, (newPath, vert, horiz)))
                            # frontierFrom.put((totalCost, newPath))

    print("function ended error")
    return [-1, None]


if __name__ == "__main__":
    global totalTime
    global maxTime

    # for all 4 tables (look at chooseWDDict for more info)
    global vertWDRanks1
    global vertWDRanks2
    global vertWDRanks3
    global vertWDRanks4

    # for A* search
    global explored
    # for bidirectional search:
    global exploredTo
    global exploredFrom

    # types of tests
    t = ["test", "single", "profile", "sandbox"]

    ###########################################################################
    ###########################################################################
    # NOTE: basic parameters
    # uses STATES (definbelow)
    HARD_CODE = False
    # generates random state (vs shuffled state)
    RANDOM = True
    # how many scambles (only for RANDOM = False)
    numScrambles = 1000 # scrambles
    TYPE = t[0] # type of run
    numTests = 100 # how many tests
    maxTime = 100 # max seconds for each test
    global BRANCH_BOUND
    global pastCostConst
    global fromConst
    BRANCH_BOUND = 81 # where to cut off branches (past cost)
    pastCostConst = 1 # constant with which to divide past cost (only 1 gives
    #                   optimal solutions
    fromConst = 1 # at every _th pass will we expand the "from" branch
    #

    # HARD CODED STATES CURRENTLY DISABLED
    STATES =   [[10, 16, 14, 1, 13, 5, 8, 3, 4, 12, 11, 2, 15, 9, 6, 7],
                [11, 12, 14, 3, 1, 2, 4, 8, 16, 13, 7, 5, 15, 10, 9, 6],
                [6, 9, 7, 15, 13, 5, 4, 16, 10, 14, 1, 11, 2, 3, 12, 8],
                [13, 14, 16, 15, 12, 5, 11, 9, 6, 4, 10, 2, 1, 8, 7, 3],
                [12, 2, 9, 13, 14, 15, 4, 10, 5, 6, 8, 16, 1, 7, 11, 3],
                [12, 8, 14, 13, 10, 2, 1, 15, 9, 11, 16, 3, 4, 6, 7, 5],
                [15, 9, 7, 6, 10, 16, 14, 4, 2, 3, 12, 13, 5, 8, 11, 1],
                [9, 4, 8, 3, 2, 11, 13, 5, 6, 7, 10, 1, 12, 16, 15, 14],
                [16, 10, 7, 1, 6, 2, 4, 13, 15, 5, 11, 3, 12, 8, 9, 14],
                [12, 1, 6, 10, 4, 5, 8, 9, 3, 13, 2, 14, 15, 7, 16, 11],
                [10, 6, 14, 1, 15, 3, 9, 2, 8, 4, 12, 7, 16, 5, 11, 13],
                [14, 9, 2, 11, 10, 6, 15, 1, 13, 8, 5, 7, 16, 12, 4, 3],
                [9, 13, 7, 3, 1, 2, 10, 6, 12, 15, 5, 14, 16, 4, 8, 11],
                [4, 10, 8, 3, 5, 14, 13, 2, 11, 6, 9, 16, 12, 1, 15, 7],
                [10, 16, 11, 13, 8, 5, 9, 15, 6, 1, 3, 14, 2, 7, 4, 12],
                [8, 14, 9, 15, 7, 6, 3, 2, 16, 13, 10, 11, 5, 12, 1, 4],
                [5, 15, 4, 6, 10, 2, 14, 11, 7, 16, 8, 9, 12, 3, 13, 1],
                [13, 3, 6, 16, 12, 7, 10, 1, 2, 11, 5, 4, 8, 9, 15, 14],
                [4, 10, 16, 3, 12, 2, 9, 14, 8, 13, 7, 6, 15, 11, 1, 5],
                [9, 3, 10, 5, 16, 13, 6, 14, 4, 2, 12, 15, 7, 1, 8, 11],
                [14, 15, 4, 12, 16, 8, 7, 1, 13, 2, 10, 6, 3, 5, 11, 9],
                [14, 1, 8, 4, 7, 5, 2, 12, 11, 13, 16, 6, 3, 9, 10, 15],
                [9, 12, 4, 3, 11, 10, 7, 16, 14, 15, 13, 6, 5, 2, 8, 1],
                [12, 11, 2, 7, 14, 6, 10, 3, 15, 9, 13, 16, 4, 8, 5, 1],
                [11, 2, 6, 15, 1, 9, 14, 8, 16, 12, 4, 5, 3, 13, 10, 7],
                [11, 3, 7, 2, 16, 10, 9, 14, 12, 13, 1, 4, 5, 6, 8, 15],
                [11, 2, 15, 3, 1, 14, 8, 16, 7, 9, 13, 4, 12, 5, 10, 6],
                [3, 14, 12, 6, 11, 7, 10, 2, 9, 13, 15, 4, 16, 1, 5, 8],
                [11, 7, 6, 8, 5, 4, 14, 10, 12, 15, 13, 1, 3, 2, 9, 16],
                [4, 11, 6, 2, 9, 12, 13, 10, 3, 14, 1, 5, 15, 16, 7, 8],
                [5, 14, 7, 4, 16, 6, 12, 2, 15, 11, 8, 13, 1, 3, 10, 9],
                [13, 10, 7, 14, 6, 3, 8, 5, 9, 2, 1, 16, 4, 11, 12, 15],
                [8, 1, 4, 16, 14, 9, 2, 7, 6, 15, 12, 3, 5, 11, 13, 10],
                [5, 16, 7, 14, 13, 3, 11, 12, 2, 6, 15, 9, 1, 8, 10, 4],
                [5, 3, 11, 2, 9, 8, 1, 14, 16, 12, 6, 10, 15, 7, 4, 13],
                [12, 10, 4, 7, 3, 14, 2, 15, 9, 16, 1, 6, 8, 11, 5, 13],
                [3, 15, 2, 5, 6, 1, 8, 4, 12, 9, 16, 13, 14, 7, 10, 11],
                [10, 3, 16, 6, 2, 5, 4, 12, 7, 13, 14, 15, 11, 1, 9, 8],
                [16, 1, 6, 13, 15, 11, 9, 10, 2, 4, 8, 14, 5, 12, 7, 3],
                [2, 5, 13, 8, 1, 11, 10, 15, 3, 6, 7, 12, 9, 4, 16, 14],
                [5, 8, 4, 10, 15, 6, 1, 3, 9, 14, 16, 7, 2, 11, 12, 13],
                [14, 4, 1, 9, 15, 5, 16, 11, 8, 13, 6, 7, 2, 3, 10, 12],
                [5, 16, 1, 7, 9, 13, 11, 10, 6, 3, 15, 8, 12, 14, 2, 4],
                [3, 4, 13, 2, 12, 9, 1, 14, 15, 6, 11, 16, 7, 8, 5, 10],
                [7, 9, 10, 12, 6, 2, 13, 8, 4, 3, 16, 14, 5, 15, 1, 11],
                [6, 16, 15, 10, 8, 13, 2, 14, 11, 3, 4, 7, 5, 9, 12, 1],
                [2, 3, 6, 15, 1, 14, 8, 12, 16, 11, 10, 7, 4, 13, 5, 9],
                [13, 14, 6, 5, 1, 8, 2, 16, 7, 10, 9, 12, 4, 3, 11, 15],
                [7, 6, 13, 11, 3, 5, 1, 16, 8, 14, 2, 4, 9, 15, 12, 10],
                [9, 16, 4, 3, 10, 15, 8, 11, 2, 13, 7, 1, 5, 14, 12, 6],
                [12, 16, 1, 5, 3, 2, 4, 14, 8, 7, 11, 10, 13, 15, 9, 6],
                [11, 8, 9, 5, 3, 12, 13, 14, 2, 6, 10, 1, 16, 15, 4, 7],
                [1, 3, 2, 14, 16, 10, 7, 13, 9, 15, 5, 6, 8, 12, 4, 11],
                [10, 2, 1, 15, 14, 12, 8, 4, 13, 11, 6, 3, 16, 9, 5, 7],
                [10, 1, 14, 13, 16, 15, 4, 9, 2, 6, 12, 8, 3, 11, 5, 7],
                [4, 1, 13, 15, 8, 6, 3, 10, 7, 5, 9, 11, 16, 2, 12, 14],
                [4, 9, 5, 14, 3, 6, 15, 13, 11, 16, 7, 8, 12, 1, 10, 2],
                [14, 1, 15, 2, 12, 16, 11, 10, 3, 5, 7, 8, 9, 6, 13, 4],
                [8, 11, 12, 4, 7, 3, 10, 1, 9, 13, 14, 15, 6, 2, 5, 16],
                [14, 4, 13, 15, 16, 6, 3, 5, 1, 12, 11, 9, 2, 8, 10, 7],
                [1, 15, 10, 7, 8, 4, 16, 6, 5, 12, 13, 14, 11, 3, 2, 9],
                [13, 15, 14, 1, 9, 5, 7, 16, 11, 10, 2, 6, 4, 8, 3, 12],
                [12, 1, 10, 15, 2, 13, 3, 11, 16, 7, 8, 14, 4, 9, 6, 5],
                [6, 3, 16, 8, 14, 1, 12, 11, 4, 9, 15, 13, 2, 7, 5, 10],
                [5, 11, 6, 16, 12, 9, 10, 4, 8, 14, 3, 1, 2, 15, 7, 13],
                [1, 13, 15, 16, 3, 10, 6, 5, 4, 9, 12, 2, 7, 14, 11, 8],
                [6, 2, 5, 1, 11, 13, 14, 16, 3, 9, 7, 4, 10, 15, 8, 12],
                [6, 2, 16, 10, 7, 11, 14, 3, 1, 4, 9, 8, 13, 5, 12, 15],
                [15, 8, 3, 14, 5, 7, 4, 11, 16, 13, 10, 1, 2, 12, 9, 6],
                [11, 14, 15, 12, 16, 8, 3, 4, 7, 5, 6, 9, 13, 1, 2, 10],
                [4, 5, 11, 1, 15, 6, 3, 7, 14, 13, 16, 12, 8, 2, 10, 9],
                [7, 5, 6, 10, 9, 14, 16, 15, 13, 4, 2, 1, 3, 8, 11, 12],
                [10, 1, 5, 2, 14, 11, 13, 9, 16, 8, 7, 15, 3, 6, 4, 12],
                [3, 15, 4, 6, 14, 2, 9, 7, 8, 1, 11, 13, 10, 12, 16, 5],
                [10, 9, 7, 14, 11, 13, 2, 5, 15, 6, 8, 16, 12, 4, 1, 3],
                [11, 7, 6, 8, 4, 15, 2, 13, 3, 12, 1, 16, 10, 14, 9, 5],
                [16, 15, 1, 9, 4, 14, 11, 8, 6, 7, 10, 2, 13, 5, 12, 3],
                [4, 5, 1, 3, 10, 16, 14, 2, 6, 8, 15, 11, 13, 12, 9, 7],
                [6, 4, 1, 2, 10, 7, 14, 15, 16, 12, 11, 13, 5, 3, 8, 9],
                [6, 12, 7, 8, 3, 11, 5, 1, 10, 9, 14, 16, 2, 13, 4, 15],
                [13, 10, 11, 2, 4, 5, 6, 1, 8, 3, 12, 15, 9, 14, 7, 16],
                [3, 6, 11, 15, 16, 4, 10, 2, 5, 7, 13, 1, 8, 12, 14, 9],
                [13, 8, 14, 3, 10, 12, 9, 11, 2, 5, 7, 16, 6, 15, 1, 4],
                [7, 11, 6, 10, 4, 14, 1, 15, 2, 3, 9, 13, 5, 8, 12, 16],
                [5, 12, 6, 13, 2, 16, 15, 8, 10, 7, 1, 3, 14, 11, 4, 9],
                [3, 11, 1, 14, 8, 15, 4, 6, 10, 5, 9, 16, 2, 12, 13, 7],
                [16, 3, 10, 9, 7, 8, 11, 14, 15, 12, 4, 2, 1, 6, 5, 13],
                [4, 8, 16, 11, 3, 5, 14, 9, 15, 12, 7, 1, 10, 13, 6, 2],
                [10, 4, 12, 14, 9, 7, 3, 15, 6, 2, 16, 1, 8, 11, 13, 5],
                [10, 16, 3, 13, 1, 15, 11, 7, 2, 5, 6, 8, 12, 14, 4, 9],
                [6, 4, 12, 3, 1, 11, 16, 5, 8, 13, 7, 15, 2, 9, 14, 10],
                [15, 11, 7, 9, 3, 6, 12, 4, 14, 5, 10, 16, 8, 2, 1, 13],
                [3, 5, 14, 10, 1, 11, 15, 2, 12, 6, 4, 9, 7, 13, 16, 8],
                [7, 13, 4, 14, 16, 12, 3, 9, 15, 1, 10, 8, 2, 5, 6, 11],
                [7, 13, 5, 6, 11, 2, 8, 15, 9, 12, 4, 14, 10, 3, 1, 16],
                [6, 13, 9, 2, 10, 15, 7, 16, 12, 14, 8, 5, 11, 4, 3, 1],
                [6, 4, 9, 10, 5, 13, 7, 16, 15, 11, 2, 12, 8, 1, 3, 14],
                [4, 16, 15, 11, 9, 12, 10, 3, 1, 5, 14, 2, 6, 7, 8, 13],
                [7, 15, 3, 11, 12, 4, 8, 6, 9, 13, 10, 1, 14, 2, 5, 16],
                [11, 10, 9, 7, 4, 2, 8, 15, 12, 3, 16, 6, 1, 13, 14, 5]]

    ###########################################################################
    ###########################################################################

    # instantiate other things
    global GOAL_STATE
    GOAL_STATE = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    # load tables from pickle
    # print("Loading TABLES from pickle...")
    pic_begin = time.time()
    pickle_in = open("TABLES","rb")
    TABLES = pickle.load(pickle_in)
    pic_end = time.time()
    vertWDRanks1 = TABLES[0]
    vertWDRanks2 = TABLES[1]
    vertWDRanks3 = TABLES[2]
    vertWDRanks4 = TABLES[3]
    # print("Process time: " + str(pic_end-pic_begin) + "\n")

    print("\nConstant with which to divide pastCost: {}\n".format(pastCostConst))
    # how much to divide the past cost by in biD search

    # totalEnd = totalStart is total time for entire program
    totalStart = time.time()
    #######################################################################
    if TYPE=="test":
        # like: 5, 30, 100, failed
        seg_times = [[], [], [], []]
        # stats for end (only for solved states)
        numCorrect = 0
        solvedTimes = []
        pathLengths = []

        for i in range(numTests):
            if i == 0:
                perc = 100
            else:
                perc = (numCorrect / i)*100
            print("running test {} / {}   ({} / {} found so far... {}%)".format(str(i+1), str(numTests), str(numCorrect), str(i), perc))

            # Make a random state.
            state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            goal = copy.deepcopy(state)

            if RANDOM:
                # print("creating random state")
                random.shuffle(state)
                while not isSolvable(state):
                    random.shuffle(state)
            else:
                state = scrambler(state, numScrambles)
                # print("created "+ str(n) +"-scrambled state")

            if HARD_CODE:
                # if wish to use hard-coded test states (for debugging)
                state = STATES[i]

            print15(state)
            print("has rank " + str(rankPerm(state)))
            # [runTime, path] = AStar([state], neighbors, isGoal, doNothing, heuristicWD)
            [runTime, path] = bidirectional([state], [goal], neighbors, isGoal, doNothing, heuristicWD)
            # add general statistics
            if path != None:
                pathLengths.append(len(path))
                solvedTimes.append(runTime)
                numCorrect += 1
                print("Path Length: {} ".format(len(path)))

            # add things to seg_times
            if runTime < 0:
                seg_times[3].append(runTime)
            if 0 < runTime <=5:
                seg_times[0].append(runTime)
            if 5 < runTime <=30:
                seg_times[1].append(runTime)
            if 30 < runTime <= maxTime + 20:
                seg_times[2].append(runTime)
            print("Run Time : {}".format(runTime))

        print("\nNum Unsolved: {}\n".format(len(seg_times[3])))
        print("Num Under 5 secs: ", len(seg_times[0]))
        print("Num Under 30 secs: ", len(seg_times[0]) + len(seg_times[1]))
        print("Num Under 100 secs: ", len(seg_times[0]) + len(seg_times[1]) + len(seg_times[2]))

        if len(pathLengths) > 0:
            print("\nMedian path length: {}".format(median(pathLengths)))

        totalEnd = time.time()
        print("\nTotal Time: {}".format(totalEnd - totalStart))

        print(solvedTimes)

        n, bins, patches = plt.hist(solvedTimes, 20, color="pink")
        plt.xlabel('Solve Times')
        plt.ylabel('Frequences')
        plt.title('Frequency of Solve Times')
        plt.grid(True)
        plt.show()

        n, bins, patches = plt.hist(pathLengths, 20, color="blue")
        plt.xlabel('Path Lengths')
        plt.ylabel('Frequences')
        plt.title('Frequency of Path Lengths')
        plt.grid(True)
        plt.show()

    #######################################################################
    elif TYPE=="single":
        exploredTo = {}
        explored = {}
        exploredFrom = {}
        # Make a random state.
        state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        goal = copy.deepcopy(state)

        if RANDOM:
            random.shuffle(state)
            while not isSolvable(state):
                random.shuffle(state)
        else:
            state = scrambler(state, numScrambles)

        print15(state)
        print("has rank " + str(rankPerm(state)))

        # [runTime, path] = AStar([state], neighbors, isGoal, doNothing, heuristicWD)
        [runTime, path] = bidirectional([state], [goal], neighbors, isGoal, doNothing, heuristicWD)

        print15s(path)
        print("runTime: ", runTime)

    #######################################################################
    elif TYPE=="profile":
        print("TYPE = Profile")

        exploredTo = {}
        explored = {}
        exploredFrom = {}
        # Make a random state.
        state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        goal = copy.deepcopy(state)

        if RANDOM:
            # print("creating random state")
            random.shuffle(state)
            while not isSolvable(state):
                random.shuffle(state)
        else:
            state = scrambler(state, numScrambles)
            # print("created "+ str(n) +"-scrambled state")

        # for 100 second profile, make unsolvable
        x = state[-1]
        y = state[-2]
        state[-2] = x
        state[-1] = y

        print("starting profile...")
        cProfile.run("bidirectional([state], [goal], neighbors, isGoal, doNothing, heuristicWD)")

    #######################################################################
    elif TYPE=="sandbox":
        print("TYPE = Sandbox")
    else:
        print("RUN TYPE ERROR")
