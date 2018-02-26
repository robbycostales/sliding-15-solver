# Date: 2018-02-25
# Author: Robby Costales

# Purpose: Contains all of the main solving algorithms

import sys
import random
import ast
from queue import PriorityQueue
import time
import copy
import pickle
# local
import createTables as ct
import funcs as fu



def isGoal(state):
    """
    Checks if given state is a goal state for S15

    Args:
        state : 1d python list representation of S15
    Returns:
        boolean : True of state is goal, False otherwise
    """
    return state == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]


def neighbors(state):
    """
    Finds neighbors of an S15 state

    Args:
        state : 1d python list representation of S15
    Returns:
        list : list of neighbors (where each neighbor in given state format)
    """
    neighborhood = []

    # find blank position
    i = state.index(16)

    # move blank left?
    if i % 4 != 0:
        newState = state[:i-1] + [state[i],state[i-1]] + state[i+1:]
        neighborhood.append(newState)

    # move blank right?
    if i % 4 != 3:
        newState = state[:i] + [state[i+1],state[i]] + state[i+2:]
        neighborhood.append(newState)

    # move blank up?
    if i > 3:
        newState = state[:i-4] + [state[i]] + state[i-3:i] + [state[i-4]] + state[i+1:]
        neighborhood.append(newState)

    # move blank down?
    if i < 12:
        newState = state[:i] + [state[i+4]] + state[i+1:i+4] + [state[i]] + state[i+5:]
        neighborhood.append(newState)

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
                boo, rank = fu.stateInDict(i, sExplored)
                if not boo:
                    newNeighbors.append(i)
            neighborList = newNeighbors

            num = len(neighborList)
            nextNeighbor = neighborList[random.randint(0, num-1)]
            state = nextNeighbor

            boo, rank = fu.stateInDict(state, sExplored)
            sExplored[str(rank)] = 1
    except:
        scrambler(state, n)
    return state


def levelInput():
    """
    Asks for user input for a state (in form of list as string) and converts
    to list

    Returns:
        state that user input
    """
    string = input("Please enter the a list corresponding to the level:\n    ")
    state = ast.literal_eval(string)
    print(state)
    return state


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


def heuristicManhattan(state):
    """
    Heuristic that tells us for each tile, how far away it is from it's
    original position. We take the sum of these measurements

    Args:
        state :  list of 1d python list representations of S15
    Returns:
        heuristic (int)
    """

    total = 0
    for row in range(4):
        for col in range(4):
            index = 4*row + col
            correct = index+1
            incorrect = state[index]
            if incorrect == 0: continue
            incorrectRow = int((incorrect-1)/4)
            incorrectCol = (incorrect-1)%4
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
        rank, state (2d list)

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
        try:
            goal[goal.index(16)] = 0
        except:
            pass
        goal = fu.unFlatten(goal)

    # converting 1-list to 2-list
    conv = fu.unFlatten(state)

    if orientation == "horiz":
        goal = fu.transpose(goal)
        conv = fu.transpose(conv)

    # check intersection in each row, create 1-D list to rank
    ints = []
    for i in conv:
        for j in goal:
            ints.append(fu.numInCommon(i, j))

    # find rank of the WD state created
    rank = fu.rankPerm(ints)

    return rank, fu.unFlatten(ints)


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
        goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    vertRank, m1 = convertWD(state, goal, "vert")
    horizRank, m2 = convertWD(state, goal, "horiz")

    vertWDRanks = chooseWDDict(goal, ori="vert")
    try:
        x = vertWDRanks[str(vertRank)]
    except:
        print("vert", m1)
        x = 35

    vertWDRanks = chooseWDDict(goal, ori="horiz")
    try:
        y = vertWDRanks[str(horizRank)]
    except:
        print("horiz", m2)
        y = 35

    return x+y


def isSolvable(state):
    """
    Checks if given S15 state is solvable

    Args:
        state : 1d form
    Returns:
        boolean
    """
    # find blank position
    z = state.index(16)

    invs = 0
    for i in range(15):
        if i == z: continue
        for j in range(i+1,16):
            if j == z: continue
            if state[i] > state[j]:
                invs += 1

    return (i//4 + invs) % 2 == 1


def doNothing(path):
    """
    Visit function for a path that literally does nothing...
    """
    pass


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
        explored[str(fu.rankPerm(s))] = 1

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
                boo, rank = fu.stateInDict(neighbor, explored)
                if not boo:
                    explored[str(rank)] = 1
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    futureCost = heuristicFn(neighbor)
                    totalCost = pastCost + futureCost
                    frontier.put((totalCost, newPath))

    return [-1, None]


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
    state = fu.unFlatten(state)
    if ori == "horiz":
        state = fu.transpose(state)
    # no need to transpose, vertical MEANS ROWS!!!! not columns
    # this is confusing af
    # state = fu.transpose(state)
    for i in range(4):
        if 16 in state[i] or 0 in state[i]:
            typ = i+1
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
    startTime = time.time()

    # paths to goal state
    frontierTo = PriorityQueue()
    # paths from goal state
    frontierFrom = PriorityQueue()

    # put initial state's path in TO queue
    for s in S:
        frontierTo.put((0, [s]))
        exploredTo[str(fu.rankPerm(s))] = 1

    # put initial state's path in TO queue
    for g in G:
        frontierFrom.put((0, [g]))
        exploredFrom[str(fu.rankPerm(g))] = 1

    while frontierFrom.qsize() > 0 and frontierTo.qsize() > 0:
        # print(frontierTo.qsize(), frontierFrom.qsize(), end="\r")

        # check time
        currentTime = time.time()
        if currentTime - startTime > maxTime:
            return [-1, None]

        ### TO ------->>>>>
        # pull from TO queue
        (_, path) = frontierTo.get()
        node = path[-1]

        # check if node in other dictionary
        boo, rank = fu.stateInDict(node, exploredFrom)
        if boo:
            # have our final pathTo
            pathTo = path
            # find the pathFrom that has this overlapping node
            while frontierFrom.qsize() > 0:
                (_, pathCheck) = frontierFrom.get()
                for i in range(len(pathCheck)):
                    if rank == fu.rankPerm(pathCheck[i]):
                        # we found our pathFrom
                        pathFrom = pathCheck
                        # need to merge these paths on i (i cant appear in both)
                        finalPath = pathTo + pathFrom[i+1::-1]
                        visitFn(finalPath)
                        currentTime = time.time()
                        return [currentTime - startTime, finalPath]
        else:
            neighborhood = neighborhoodFn(node)
            for neighbor in neighborhood:
                boo, rank = fu.stateInDict(neighbor, exploredTo)
                if not boo:
                    exploredTo[str(rank)] = 1
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    # goal = None means distance to normal goal state
                    futureCost = heuristicFn(neighbor, goal=None)
                    totalCost = pastCost + futureCost
                    frontierTo.put((totalCost, newPath))

        ### <<<<<------- FROM
        # pull from FROM queue
        (_, path) = frontierFrom.get()
        node = path[-1]

        # check if node in other dictionary
        boo, rank = fu.stateInDict(node, exploredTo)
        if boo:
            # have our final pathFrom
            pathFrom = path
            # find the pathTo that has this overlapping node
            while frontierTo.qsize() > 0:
                (_, pathCheck) = frontierTo.get()
                for i in range(len(pathCheck)):
                    if rank == fu.rankPerm(pathCheck[i]):
                        # we found our pathTo
                        pathTo = pathCheck
                        # need to merge these paths on i (i cant appear in both)
                        finalPath = pathTo[:i] + pathFrom[::-1]
                        visitFn(finalPath)
                        currentTime = time.time()
                        return [currentTime - startTime, finalPath]
        else:
            neighborhood = neighborhoodFn(node)
            for neighbor in neighborhood:
                boo, rank = fu.stateInDict(neighbor, exploredFrom)
                if not boo:
                    exploredFrom[str(rank)] = 1
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    # goal = S[0], means distance to initial state
                    futureCost = heuristicFn(neighbor, goal=S[0])
                    totalCost = pastCost + futureCost
                    frontierFrom.put((totalCost, newPath))
    return [-1, None]



if __name__ == "__main__":
    RANDOM = False
    # number of scrambles
    N = 40
    TEST = True
    numTests = 100

    global maxTime
    # for one way search:
    global explored
    # for all 4 tables (look at chooseWDDict for more info)
    global vertWDRanks1
    global vertWDRanks2
    global vertWDRanks3
    global vertWDRanks4
    # for bidirectional search:
    global exploredTo
    global exploredFrom

    # load tables from pickle
    print("Loading TABLES from pickle...")
    pic_begin = time.time()
    pickle_in = open("TABLES","rb")
    TABLES = pickle.load(pickle_in)
    pic_end = time.time()
    vertWDRanks1 = TABLES[0]
    vertWDRanks2 = TABLES[1]
    vertWDRanks3 = TABLES[2]
    vertWDRanks4 = TABLES[3]
    print("Process time: " + str(pic_end-pic_begin))


    if TEST:
        # like: 5, 30, 100, failed
        timez = [[], [], [], []]
        for i in range(numTests):
            print("test: " + str(i+1) + " / 100", end = "\r")

            exploredTo = {}
            explored = {}
            exploredFrom = {}

            maxTime = 100
            # Make a random state.
            state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            goal = copy.deepcopy(state)

            if RANDOM:
                # print("creating random state")
                random.shuffle(state)
                while not isSolvable(state):
                    random.shuffle(state)
            else:
                state = scrambler(state, N)
                # print("created "+ str(n) +"-scrambled state")

            # [runTime, path] = AStar([state], neighbors, isGoal, doNothing, heuristicWD)
            [runTime, path] = bidirectional([state], [goal], neighbors, isGoal, doNothing, heuristicWD)


            if runTime < 0:
                timez[3].append(runTime)
            if 0 < runTime <=5:
                timez[0].append(runTime)
            if 5 < runTime <=30:
                timez[1].append(runTime)
            if 30 < runTime <=101:
                timez[2].append(runTime)

        # print("\nUGLY LIST: ")
        # print(timez)

        print("\nNum Unsolved: ", len(timez[3]))
        print("Num Under 5 secs: ", len(timez[0]))
        print("Num Under 30 secs: ", len(timez[0]) + len(timez[1]))
        print("Num Under 100 secs: ", len(timez[0]) + len(timez[1]) + len(timez[2]))


    else:
        exploredTo = {}
        explored = {}
        exploredFrom = {}

        maxTime = 100
        # Make a random state.
        state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        goal = copy.deepcopy(state)

        if RANDOM:
            print("creating random state")
            random.shuffle(state)
            while not isSolvable(state):
                random.shuffle(state)
        else:
            state = scrambler(state, N)
            print("created "+ str(N) +"-scrambled state")


        print15(state)
        print("has rank " + str(fu.rankPerm(state)))

        # [runTime, path] = AStar([state], neighbors, isGoal, doNothing, heuristicWD)
        [runTime, path] = bidirectional([state], [goal], neighbors, isGoal, doNothing, heuristicWD)

        print15s(path)
        print("runTime: ", runTime)
