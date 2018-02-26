# Date: 2018-02-25
# Author: Robby Costales

# Purpose: Contains many misc functions

def numInCommon(list1, list2):
    """
    How many elements in common two given lists have

    Args:
        list1
        list2
    Returns:
        int : number of elements in common
    """
    return len(list(set(list1).intersection(list2)))


def transpose(og):
    """
    Tranposes 2d python list

    Args:
        og : should be in the form of a 2d list
    Returns:
        transpose
    """
    return [list(x) for x in zip(*og)]


def unFlatten(state):
    """
    turns 1x16 python list to 4x4 list

    Args:
        state : either WD state or S15 state in 1x16 form
    Returns:
        state : state in 4x4 form
    """
    conv = []
    for i in range(4):
        temp = []
        for j in range(4):
            temp.append(state[4*i+j])
        conv.append(temp)
    return conv


def stateInDict(state, dictionary):
    """
    Checks if a state's rank is in given dictionary (uses rankPerm function)
    The ranking function is used for all ranks

    Args:
        state : in 1-d or 2-d list format
        dictionary
    Returns:
        boolean, rank (key of dictionary)
    """
    rank = rankPerm(state)
    if str(rank) in dictionary:
        return True, rank
    else:
        return False, rank


def rankPerm(perm, inverse = None, m = None):
    """
    rankPerm(perm) returns the rank of permutation perm.
    The rank is done according to Myrvold, Ruskey "Ranking and unranking
    permutations in linear-time".
    perm should be a 1-based list, such as [1,2,3,4,5].

    However, this function will automatically flatten a 2d array into a
    1-based list
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
