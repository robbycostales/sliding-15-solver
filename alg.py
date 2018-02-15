# Sliding 15-Puzzle
import sys
import random
import ast
from queue import PriorityQueue

# python3
#import queue
#frontier = queue.PriorityQueue()

def isGoal(state):
    return state == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

def neighbors(state):
    neighborhood = []

    # find blank position
    i = state.index(0)

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


# TODO: don't regenerate previously generated states
def scrambler(state, n):
    for step in range(n):
        neighborList = neighbors(state)
        num = len(neighborList)
        nextNeighbor = neighborList[random.randint(0, num-1)]
        state = nextNeighbor
    return state


def levelInput():
    string = input("Please enter the a list corresponding to the level:\n    ")
    state = ast.literal_eval(string)

    print(state)

    return state


def heuristicBad(state):
    return 0


def heuristicMedium(state):
    solution = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
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

def AStar(S, neighborhoodFn, goalFn, visitFn, heuristicFn):

    frontier = PriorityQueue()
    for s in S:
        frontier.put((0, [s]))

    while frontier.qsize() > 0:
        (_, path) = frontier.get()
        node = path[-1]

        # debugging
        if len(path) > 100:
            print15s(path)
            return

        if goalFn(node):
            visitFn(path)
            return
        else:
            neighborhood = neighborhoodFn(node)
            for neighbor in neighborhood:
                if neighbor not in path:
                    newPath = path + [neighbor]
                    pastCost = len(newPath)-1
                    futureCost = heuristicFn(neighbor)
                    totalCost = pastCost + futureCost
                    frontier.put((totalCost, newPath))




if __name__ == "__main__":
    # Make a random state.
    # state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
    # state = scrambler(state, 150)

    state = levelInput()
    print15(state)

    AStar([state], neighbors, isGoal, print15s, heuristicGood)
