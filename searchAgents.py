# searchAgents.py
# ---------------
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
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import math
import itertools


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActionSequence(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class DeceptiveSearchAgentpi2(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        # COMP90054 Task 4 - Implement your deceptive search algorithm here
        #util.raiseNotDefined()
        self.actions = []
        currentState = state
        start_node = currentState.getPacmanPosition()
        goal_real = currentState.getFood().asList()[0]
        goal_fake = currentState.getCapsules()[0]

        # create three problems for path a, b and c
        a_problem = PositionSearchProblem(state,start=start_node, goal=goal_real, warn=False, visualize=False)
        b_problem = PositionSearchProblem(state, start=start_node, goal=goal_fake, warn=False, visualize=False)
        c_problem = PositionSearchProblem(state, start=goal_fake, goal=goal_real, warn=False, visualize=False)
        # find the path action of a, b and c
        a_path = search.aStarSearch(a_problem,heuristic=manhattanHeuristic)
        b_path = search.aStarSearch(b_problem, heuristic=manhattanHeuristic)
        c_path = search.aStarSearch(c_problem, heuristic=manhattanHeuristic)
        # calculate the Beta point and y
        Beta =int((len(c_path)+ len(a_path) - len(b_path))/2)
        y_length = len(c_path) - Beta
        y_path = []
        for i in range(y_length):
            y_path.append(c_path[i])
        # find the point starting from the fake goal_real
        x, y =goal_fake
        for action in y_path:
            if action == Directions.NORTH:
                y += 1
            elif action == Directions.SOUTH:
                y -= 1
            elif action == Directions.EAST:
                x += 1
            elif action == Directions.WEST:
                x -= 1
        point_between = (x,y)
        # get two part of action and add together
        d2_problem = PositionSearchProblem(state, start=start_node, goal=point_between, warn=False, visualize=False)
        beta_problem = PositionSearchProblem(state, start=point_between, goal=goal_real, warn=False, visualize=False)
        action_1 = search.aStarSearch(d2_problem,manhattanHeuristic)
        action_2 = search.aStarSearch(beta_problem, manhattanHeuristic)
        self.actions = action_1+action_2
        return self.actions


# create a global dictionary to store the real goal and fake goal
global dict
dict = {}
class DeceptiveSearchAgentpi3(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        # COMP90054 Task 4 - Implement your deceptive search algorithm here
        self.actions = []
        currentState = state
        start_node = currentState.getPacmanPosition()
        goal_real = currentState.getFood().asList()[0]
        goal_fake = currentState.getCapsules()[0]

        # create three problems for path a, b and c
        a_problem = PositionSearchProblem(state, start=start_node, goal=goal_real, warn=False, visualize=False)
        b_problem = PositionSearchProblem(state, start=start_node, goal=goal_fake, warn=False, visualize=False)
        c_problem = PositionSearchProblem(state, start=goal_fake, goal=goal_real, warn=False, visualize=False)
        # find the path action of a, b and c
        a_path = search.aStarSearch(a_problem, heuristic=manhattanHeuristic)
        b_path = search.aStarSearch(b_problem, heuristic=manhattanHeuristic)
        c_path = search.aStarSearch(c_problem, heuristic=manhattanHeuristic)
        # calculate the Beta point and y
        Beta = int((len(c_path) + len(a_path) - len(b_path)) / 2)
        y_length = len(c_path) - Beta
        y_path = []
        for i in range(y_length):
            y_path.append(c_path[i])
        # find the point starting from the fake goal_real
        x, y = goal_fake
        for action in y_path:
            if action == Directions.NORTH:
                y += 1
            elif action == Directions.SOUTH:
                y -= 1
            elif action == Directions.EAST:
                x += 1
            elif action == Directions.WEST:
                x -= 1
        point_between = (x, y)
        # get two part of action and add together

        global dict
        dict['goal_real'] = goal_real
        dict['goal_fake'] = goal_fake
        d3_problem = PositionSearchProblem(state, start=start_node, goal=point_between, warn=False, visualize=False)
        beta_problem = PositionSearchProblem(state, start=point_between, goal=goal_real, warn=False, visualize=False)
        action_1 = search.aStarSearch(d3_problem, pid3Heuristic)
        action_2 = search.aStarSearch(beta_problem, manhattanHeuristic)
        self.actions = action_1 + action_2
        return self.actions
# the heuristic for pid3
def pid3Heuristic(position, problem, info={}):
    alpha = 4
    xy1 = position
    xy2 = problem.goal
    if 'goal_real' in dict:
        goal_real = dict['goal_real']
    if 'goal_fake' in dict:
        goal_fake = dict['goal_fake']
    if abs(xy1[0] - goal_real[0]) + abs(xy1[1] - goal_real[1]) < abs(xy1[0] - goal_fake[0]) + abs(xy1[1] - goal_fake[1]):
        return alpha * (abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1]))
    else:
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, child
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1, 1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE


    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display):  # @UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist)  # @UndefinedVariable

        return isGoal

    def expand(self, state):
        """
        Returns child states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (child, action, stepCost), where 'child' is a
         child to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that child
        """

        children = []
        for action in self.getActions(state):
            nextState = self.getNextState(state, action)
            cost = self.getActionCost(state, action, nextState)
            children.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return children

    def getActions(self, state):
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(state, action), (
            "Invalid next state passed to getActionCost().")
        return self.costFn(next_state)

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        return (nextx, nexty)

    def getCostOfActionSequence(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)


class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """

    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and child function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0  # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def expand(self, state):
        """
        Returns child states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (child,
            action, stepCost), where 'child' is a child to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that child
        """

        children = []
        for action in self.getActions(state):
            # Add a child state to the child list if the action is legal
            # You should call getActions, getActionCost, and getNextState.
            "*** YOUR CODE HERE ***"

        self._expanded += 1  # DO NOT CHANGE
        return children

    def getActions(self, state):
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(state, action), (
            "Invalid next state passed to getActionCost().")
        return 1

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state[0]
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        # you will need to replace the None part of the following tuple.
        return ((nextx, nexty), None)

    def getCostOfActionSequence(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x, y = self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    return 0  # Default to trivial solution


class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid, capsules ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
      capsules:       a tuple containing tuples (x,y) that specify the location of each capsule
    """

    def __init__(self, startingGameState):
        # Note our starting state now includes the Capsule positions
        self.start = (
        startingGameState.getPacmanPosition(), startingGameState.getFood(), tuple(startingGameState.getCapsules()))
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {}  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def expand(self, state):
        "Returns child states, the actions they require, and a cost of 1."
        children = []
        self._expanded += 1  # DO NOT CHANGE
        for action in self.getActions(state):
            next_state = self.getNextState(state, action)
            action_cost = self.getActionCost(state, action, next_state)
            children.append((next_state, action, action_cost))
        return children

    def getActions(self, state):
        possible_directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(state, action), (
            "Invalid next state passed to getActionCost().")
        dx, dy = Actions.directionToVector(action)
        x, y = state[0]
        x, y = int(x + dx), int(y + dy)
        position = x, y
        if (position in list(state[2])):
            return 0
        else:
            return 1

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state[0]
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        nextCapsules = list(state[2])
        position = nextx, nexty
        if position in nextCapsules:
            nextCapsules.remove(position)
        nextCapsules = tuple(nextCapsules)
        # Our new state information now contains Capsule locations
        return ((nextx, nexty), nextFood, nextCapsules)

    def getCostOfActionSequence(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        state = self.startingGameState
        x, y = self.getStartState()[0]
        capsules = list(self.getStartState()[2])
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            position = x, y
            if (position in capsules):
                capsules.remove(position)
            else:
                cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def get_distance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)


def get_min(queue):
    cost = 9999999
    cheapest = -1
    for index, item in enumerate(queue):
        if item[1] < cost:
            cost = item[1]
            cheapest = index
    cheapestItem = queue[cheapest]
    del queue[cheapest]
    return cheapestItem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be admissible to ensure correctness.

    The state is a tuple ( pacmanPosition, foodGrid, capsules) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.  capsules contains a tuple of capsule locations.

    If you want access to info like walls, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid, capsules = state
    # COMP90054 Task 3, Implement your code here
    "*** YOUR CODE HERE ***"
    foods = foodGrid.asList()
    # returns 0 at every goal state, which there is no food left
    if len(foods) == 0:
        return 0
    if 'num_capsules' not in problem.heuristicInfo:
        problem.heuristicInfo['num_capsules'] = len(capsules)
    eaten_capsules = False
    if len(capsules) < problem.heuristicInfo['num_capsules']:
        eaten_capsules = True
    # update the capsules numbers
    problem.heuristicInfo['num_capsules'] = len(capsules)
    # 1. number of food left
    remaining_foods = len(foods)

    # remember the number of foods in the beginning
    if 'foodsnumbers' not in problem.heuristicInfo:
        problem.heuristicInfo['foodsnumbers'] = remaining_foods
    #
    # 2. number of capsules left
    remaining_capsules = len(capsules)

    # #
    # # 3. maze distances to the closest food
    # min_step_food = 0
    # for food in foods:
    #     if (position, food) not in problem.heuristicInfo or eaten_capsules:
    #         problem.heuristicInfo[(position, food)] = my_mazeDistance(position, food, problem.startingGameState, capsules)
    #
    #     if min_step_food > problem.heuristicInfo[(position, food)] or min_step_food == 0:
    #         min_step_food = problem.heuristicInfo[(position, food)]
    #
    # # 4. maze distances to the farthest food
    # max_step_food = 0
    # for food in foods:
    #     if (position, food) not in problem.heuristicInfo:
    #          problem.heuristicInfo[(position,food)] = my_mazeDistance(position,
    #                                                                   food,
    #                                                                   problem.startingGameState,
    #                                                                   capsules)
    #     if max_step_food < problem.heuristicInfo[(position,food)]:
    #         max_step_food = problem.heuristicInfo[(position,food)]
    #
    #
    # # 5.the largest distance from food to food
    # f2f_max_distance = 0
    # for food1 in foods:
    #     for food2 in foods:
    #         if (food1, food2) not in problem.heuristicInfo:
    #             problem.heuristicInfo[(food1, food2)] = my_mazeDistance(food1,
    #                                                                     food2,
    #                                                                     problem.startingGameState,
    #                                                                     capsules)
    #         if f2f_max_distance < problem.heuristicInfo[(food1, food2)]:
    #             f2f_max_distance = problem.heuristicInfo[(food1, food2)]
    # # 6. longest distance among all points
    # longest_distance = 0
    # longest_distances = []
    # longest_distances.append(position)
    # for food in foods:
    #     longest_distances.append(food)
    # for point1 in longest_distances:
    #     for point2 in longest_distances:
    #         if point1 != point2:
    #             if (point1, point2) not in problem.heuristicInfo:
    #                 problem.heuristicInfo[(point1, point2)] = my_mazeDistance(point1,
    #                                                                     point2,
    #                                                                     problem.startingGameState,
    #                                                                     capsules)
    #             if longest_distance < problem.heuristicInfo[(point1, point2)]:
    #                 longest_distance = problem.heuristicInfo[(point1, point2)]

    # 7. distance starting from position and go over all the food permutations, h is the smallest distance
    # the running time is very long.
    # test 13 needs very long time as well,
    # there are 31 foods in that test case 13 resulting in 31! kinds of permutations
    # (not able to compute, but return some remaining_foods numbers can speed up the process)
    if remaining_foods <= problem.heuristicInfo['foodsnumbers'] * 0.85:
        # return 10% remaining_foods to speed up the test
        # if remaining_foods <= problem.heuristicInfo['foodsnumbers']:
        all_distance = 99999  # a large number to be minimised
        if tuple([position] + foods) in problem.heuristicInfo:
            return problem.heuristicInfo[tuple([position] + foods)]
        for points in itertools.permutations(foods):
            current_position = position
            temp_distance = 0
            for poi in points:
                if ((poi, current_position) not in problem.heuristicInfo and \
                        (current_position, poi) not in problem.heuristicInfo):
                # if ((poi, current_position) not in problem.heuristicInfo and
                #     (current_position, poi) not in problem.heuristicInfo) or \
                #         eaten_capsules:
                    # uncomment this condition can reduce the expanded nodes
                    # but the running time will be increased exponentially
                    # the time complexity will be increased exponentially as the number of food increased
                    d = my_mazeDistance(poi, current_position, problem.startingGameState, capsules)
                    problem.heuristicInfo[(poi, current_position)] = d
                    temp_distance += d
                else:
                    if (poi, current_position) in problem.heuristicInfo:
                        temp_distance += problem.heuristicInfo[(poi, current_position)]
                    elif (current_position, poi) in problem.heuristicInfo:
                        temp_distance += problem.heuristicInfo[(current_position, poi)]
                current_position = poi
            all_distance = min(temp_distance, all_distance)
        problem.heuristicInfo[tuple([position] + foods)] = all_distance
        return all_distance  # 1006
    else:
        return remaining_foods

    # # 8. distance starting from position and to closest food, than from closest food to farthest food
    # #(find by manhattan), find the position to the closest food
    # temp_position = position
    # distance_closest_food_manh = 99999
    # for food in foods:
    #     if util.manhattanDistance(position,food) <= distance_closest_food_manh:
    #         distance_closest_food_manh = util.manhattanDistance(position,food)
    #         temp_position = food
    # closest_food = temp_position
    # # find the distance from closest food to the farthest food
    # farthest_food = temp_position
    # distance_cloest_To_farthest_food = 0
    # for food in foods:
    #     if util.manhattanDistance(temp_position, food) >= distance_cloest_To_farthest_food:
    #         distance_cloest_To_farthest_food = util.manhattanDistance(temp_position, food)
    #         farthest_food = food
    #
    # distance_closest_food_maze = my_mazeDistance(position,closest_food,problem.startingGameState,capsules)
    # distance_farthest_food_maze = my_mazeDistance(closest_food,farthest_food,problem.startingGameState,capsules)
    #
    # return distance_farthest_food_maze + distance_closest_food_maze #2383, but more reasonable then 7
    ##################################################
    # test zones
    ##################################################


def my_mazeDistance(point1, point2, gameState, capsules):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    cost = search.My_Search(prob, capsules)
    # print('______________________________')
    # print(position)
    # print('cap=======================')
    # print(capsules)
    # count the capsules on the way
    # num_capsule = 0
    # for cap in capsules:
    #     if cap in position:
    #         num_capsule += 1
    return cost


class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"

    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while (currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState)  # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateChild(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    child function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x, y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
