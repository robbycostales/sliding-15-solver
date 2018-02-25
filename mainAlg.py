# Sliding 15-Puzzle
import sys
import random
import ast
from queue import PriorityQueue
import time
import createTables as cT

# python3
#import queue
#frontier = queue.PriorityQueue()

def isGoal(state):
    return state == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

def transpose(og):
    return [list(x) for x in zip(*og)]

def unFlatten(state):
    """
    turn 1x16 list to 4x4 list
    """
    conv = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(state[4*i+j])
        conv.append(temp)
    return conv

def neighbors(state):
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
    for row in range(4):
        for col in range(4):
            if state[4*row+col] < 10:
                    sys.stdout.write(" ")
            sys.stdout.write(str(state[4*row+col]))
            sys.stdout.write("\t")
        print("")



def print15s(path):
    for i, state in enumerate(path):
        print("step " + str(i))
        print15(state)
        print("")


def scrambler(state, n):
    sExplored = {}
    for step in range(n):
        neighborList = neighbors(state)

        # # don't regenerate previously generated states
        newNeighbors = []
        for i in neighborList:
            boo, rank = rankInExplored(i, sExplored)
            if not boo:
                newNeighbors.append(i)
        neighborList = newNeighbors

        num = len(neighborList)
        nextNeighbor = neighborList[random.randint(0, num-1)]
        state = nextNeighbor

        boo, rank = rankInExplored(state, sExplored)
        sExplored[str(rank)] = 1
    return state



def levelInput():
    """
    Asks for user input for a state (in form of list as string) and converts to list
    """
    string = input("Please enter the a list corresponding to the level:\n    ")
    state = ast.literal_eval(string)
    print(state)
    return state



def heuristicBad(state):
    return 0



def heuristicMedium(state):
    solution = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    return len([i for i in range(16) if state[i] != solution[i]])



def heuristicGood(state):
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


def numInCommon(list1, list2):
    """
    returns how many elements in common these two lists have
    """
    return len(list(set(list1).intersection(list2)))


def convertWD(state, orientation="vert"):
    """
    returns rank of the WD state for vert orientation
    user orientation = horiz for horizontal (basically just tranposes things)
    """

    # zero because we want 4, 4, 4, 3 in goal rep not 4, 4, 4, 4
    # would normally be 16
    goal =  [[1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]]

    # converting 1-list to 2-list
    conv = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(state[4*i+j])
        conv.append(temp)

    if orientation == "horiz":
        goal = transpose(goal)
        conv = transpose(conv)

    # check intersection in each row, create 1-D list to rank
    ints = []
    for i in conv:
        for j in goal:
            ints.append(numInCommon(i, j))

    # find rank of the WD state created
    rank = rankPerm(ints)

    return rank, unFlatten(ints)


def heuristicWD(state):
    """
    given a state, what is the walking distance heuristic value?
    """
    vertRank, m1 = convertWD(state, "vert")
    horizRank, m2 = convertWD(state, "horiz")

    try:
        x = vertWDRanks[str(vertRank)]
    except:
        x = 35
        print("invalid dictionary key for:")
        print("vert:", m1)

    try:
        y = vertWDRanks[str(horizRank)]
    except:
        y = 35
        print("invalid dictionary key for:")
        print("horiz:", m2)

    return x+y



def rankInExplored(state, dictionary):
    """
    Checks if a state's rank is in given dictionary
    """
    rank = rankPerm(state)
    if str(rank) in dictionary:
        return True, rank
    else:
        return False, rank



def AStar(S, neighborhoodFn, goalFn, visitFn, heuristicFn):
    global maxTime
    startTime = time.time()

    frontier = PriorityQueue()

    for s in S:
        frontier.put((0, [s]))
        explored[str(rankPerm(s))] = 1

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
                boo, rank = rankInExplored(neighbor, explored)
                if str(rank) not in explored:
                    explored[str(rank)] = 1
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    futureCost = heuristicFn(neighbor)
                    totalCost = pastCost + futureCost
                    frontier.put((totalCost, newPath))

    return [-1, None]


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


    # change all 0s to 5s
    for i in range(len(perm)):
        if perm[i] == 0:
            perm[i] = 5

    # end of Robby's edits

    # return str(perm)

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



def isSolvable(state):
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
    pass



if __name__ == "__main__":
    RANDOM = True

    global maxTime
    global explored
    global vertWDRanks

    print("creating tables...")
    vertWDRanks = cT.createTables()

    explored = {}
    maxTime = 100
    # Make a random state.
    state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    if RANDOM:
        print("creating random state")
        random.shuffle(state)
        while not isSolvable(state):
            random.shuffle(state)
    else:
        n = 80
        state = scrambler(state, 80)
        print("created "+ str(n) +"-scrambled state")


    print15(state)
    print("has rank " + str(rankPerm(state)))
    [runTime, path] = AStar([state], neighbors, isGoal, doNothing, heuristicWD)
    print15s(path)
    print("runTime: ", runTime)
