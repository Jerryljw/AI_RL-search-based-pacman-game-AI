# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import math
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def expand(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (child,
        action, stepCost), where 'child' is a child to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that child.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
          state: Search state

        For a given state, this should return a list of possible actions.
        """
        util.raiseNotDefined()

    def getActionCost(self, state, action, next_state):
        """
          state: Search state
          action: action taken at state.
          next_state: next Search state after taking action.

        For a given state, this should return the cost of the (s, a, s') transition.
        """
        util.raiseNotDefined()

    def getNextState(self, state, action):
        """
          state: Search state
          action: action taken at state

        For a given state, this should return the next state after taking action from state.
        """
        util.raiseNotDefined()

    def getCostOfActionSequence(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    mystack = util.Stack()
    startNode = (problem.getStartState(), '', 0, [])
    mystack.push(startNode)
    visited = set()
    while mystack :
        node = mystack.pop()
        state, action, cost, path = node
        if state not in visited :
            visited.add(state)
            if problem.isGoalState(state) :
                path = path + [(state, action)]
                break;
            succNodes = problem.expand(state)
            for succNode in succNodes :
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                mystack.push(newNode)
    actions = [action[1] for action in path]
    del actions[0]
    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    open = util.PriorityQueue()
    initial_node = (problem.getStartState(), '', 0, [])
    open.push(initial_node,0)
    closed = set()
    while not open.isEmpty():
        sigma = open.pop()
        state, action, cost, path = sigma
        this_g = cost
        if (state not in closed) or (this_g < best_g):
            closed.add(state)
            best_g = this_g
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])

                heur = cost + succCost
                if heur < math.inf:
                    open.push(newNode, heur)
    return []

# search in task 3 for mymazeDistance
def My_Search(problem,capsules):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    open = util.PriorityQueue()
    initial_node = (problem.getStartState(), '', 0, [])
    open.push(initial_node,0)
    closed = set()
    while not open.isEmpty():
        sigma = open.pop()
        state, action, cost, path = sigma
        this_g = cost
        if (state not in closed) or (this_g < best_g):
            closed.add(state)
            best_g = this_g
            if problem.isGoalState(state):
                # path = path + [(state, action)]
                # actions = [action[1] for action in path]
                # del actions[0]
                # state = [state[0] for state in path]
                # del state[0]
                return cost
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                # count the capsules on the way and reduce the cost of it
                capsules_on_way = 0
                if succState in capsules:
                    capsules_on_way += 1
                heur = cost + succCost - capsules_on_way
                newNode = (succState, succAction, heur, path + [(state, action)])
                if heur < math.inf:
                    open.update(newNode, heur)
    return 0


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    #COMP90054 Task 1, Implement your A Star search algorithm here
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # part from sudocode on lecture slides and the given dfs code
    open = util.PriorityQueue()
    initial_node = (problem.getStartState(), '', 0, [])
    h0 = heuristic(initial_node[0],problem)
    open.push(initial_node,h0)
    closed = set()
    best_g = h0
    while not open.isEmpty():
        sigma = open.pop()
        state, action, cost, path = sigma
        this_g = heuristic(state,problem) + cost
        if (state not in closed) or (this_g < best_g):
            closed.add(state)
            best_g = this_g
            if problem.isGoalState(state):
                path = path + [(state, action)]
                actions = [action[1] for action in path]
                del actions[0]
                return actions
            succNodes = problem.expand(state)
            for succNode in succNodes:
                succState, succAction, succCost = succNode
                newNode = (succState, succAction, cost + succCost, path + [(state, action)])
                heur = cost + succCost + heuristic(succState,problem)
                if heur < math.inf:
                    open.push(newNode,heur)
    return []




        
def recursivebfs(problem, heuristic=nullHeuristic) :
    #COMP90054 Task 2, Implement your Recursive Best First Search algorithm here
    "*** YOUR CODE HERE ***"
    # part from sudocode on assignment document and the given dfs code
    def RBFS(problem, node, f_limit):
        state, action, cost, path = node[0]
        node_f = node[1]
        if problem.isGoalState(state):
            path = path + [(state, action)]
            actions = [action[1] for action in path]
            del actions[0]
            return actions, 0 # the 0 here is not important
        successors = problem.expand(state)
        # leaves nodes, no successors
        if len(successors) == 0:
            return False, math.inf
        successor_nodes = []
        for successor in successors:
            succState, succAction, succCost = successor
            # s.g + s.h
            heur = cost + succCost + heuristic(succState,problem)
            # s.f
            succ_f = max(heur, node_f)
            successor_nodes.append([(succState, succAction, cost + succCost, path + [(state, action)]), succ_f])
        while True:
            # the lowest f-value node in successors
            successor_nodes.sort(key=lambda x: x[1])
            best = successor_nodes[0]
            best_f = best[1]
            if best_f > f_limit:
                return False, best_f
            if len(successor_nodes) > 1:
                alternative = successor_nodes[1][1]
            else:
                alternative = math.inf
            result, best[1] = RBFS(problem, best, min(f_limit, alternative))
            if result is not False:
                return result, best[1] # best_f here is not important
    # initialize the node
    initial_f = 0
    initial_node = (problem.getStartState(), '', 0, [])
    # a list to represents the node and a alternative f value: [tuple, f]
    initial_node_list = [initial_node,initial_f]
    result, f = RBFS(problem, initial_node_list, math.inf)
    return result



    
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
rebfs = recursivebfs
