# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        powFood = currentGameState.getCapsules()
        pos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        tot = successorGameState.getScore()
        fmin = -1
        fdis = 0
        for food in newFood.asList():
            fdis = manhattanDistance(food,pos)
            if fmin == -1 or fdis < fmin :
                fmin = fdis
        i = 1
        gdis = []
        for g in newGhostStates:
            gpos = successorGameState.getGhostPosition(i)
            xg,yg = gpos
            gdis.append(manhattanDistance(gpos,pos))
            
            i += 1
        for o in gdis:
            if o<2:
                return -1000
        tot = sum(gdis) + 10*tot + -1.5*fmin
        return tot

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def value(self, currentGameState : GameState, d, agent):
        if agent==currentGameState.getNumAgents():
            agent = 0
            d +=1
        if currentGameState.isLose() or currentGameState.isWin() or d==self.depth:
            return self.evaluationFunction(currentGameState)
        if agent==0:
            return self.maxValue(currentGameState,d,agent)[1]
        else:
            return self.minValue(currentGameState,d,agent)[1]
        

    def maxValue(self, currentGameState : GameState, d, agent):
        v = ("max",-float("inf"))
        legalMoves = currentGameState.getLegalActions(agent)
        for action in legalMoves:
            acval = (action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1))
            if v[1] < acval[1] :
                v = acval
        return v
            
    def minValue(self, currentGameState : GameState, d, agent):
        v = ("min",float("inf"))
        legalMoves = currentGameState.getLegalActions(agent)
        for action in legalMoves:
            acval = (action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1))
            if v[1] > acval[1] :
                v = acval
        return v
                
        
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        return self.maxValue(gameState, 0, 0)[0]
    
      


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def value(self, currentGameState : GameState, d, agent, alfa, beta):
        if agent==currentGameState.getNumAgents():
            agent = 0
            d +=1
        if currentGameState.isLose() or currentGameState.isWin() or d==self.depth:
            return self.evaluationFunction(currentGameState)
        if agent==0:
            return self.maxValue(currentGameState,d,agent,alfa,beta)[1]
        else:
            return self.minValue(currentGameState,d,agent,alfa,beta)[1]
        

    def maxValue(self, currentGameState : GameState, d, agent, alfa, beta):
        v = ("max",-float("inf"))
        legalMoves = currentGameState.getLegalActions(agent)
        for action in legalMoves:
            acval = (action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1,alfa,beta))
            if v[1] < acval[1] :
                v = acval
            if v[1] > beta :
                return v
            alfa = max(alfa,v[1])
            
        return v
            
    def minValue(self, currentGameState : GameState, d, agent, alfa, beta):
        v = ("min",float("inf"))
        legalMoves = currentGameState.getLegalActions(agent)
        for action in legalMoves:
            acval = (action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1,alfa,beta))
            if v[1] > acval[1] :
                v = acval
            if v[1] < alfa :
                return v
            beta = min(beta,v[1])
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState,0,0,-float("inf"),float("inf"))[0]
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def value(self, currentGameState : GameState, d, agent):
        if agent==currentGameState.getNumAgents():
            agent = 0
            d +=1
        if currentGameState.isLose() or currentGameState.isWin() or d==self.depth:
            return self.evaluationFunction(currentGameState)
        if agent==0:
            return self.maxValue(currentGameState,d,agent)[1]
        else:
            return self.expValue(currentGameState,d,agent)[1]
        

    def maxValue(self, currentGameState : GameState, d, agent):
        v = ("max",-float("inf"))
        legalMoves = currentGameState.getLegalActions(agent)
        for action in legalMoves:
            acval = (action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1))
            if v[1] < acval[1] :
                v = acval
        return v
            
    def expValue(self, currentGameState : GameState, d, agent):
        legalMoves = currentGameState.getLegalActions(agent)
        acvals = [(action,self.value(currentGameState.generateSuccessor(agent,action),d,agent+1)) for action in legalMoves]
        o = 0
        for a in acvals:
            o += a[1]
        v = (random.choice(acvals[0]),o)
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState,0,0)[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Evaluates: 
            Closest food distance, 
            Total distance to ghosts (factors as good if ghosts scared otherwise bad), 
            Amount of power pelets left,
            Current score
    """
    "*** YOUR CODE HERE ***"
    successorGameState = currentGameState
    powFood = successorGameState.getCapsules()
    pos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    tot = successorGameState.getScore()

    fdis = [manhattanDistance(food,pos) for food in newFood.asList()]
    fo = 0
    if len(fdis)>0:
        fo = min(fdis)

    powow = len(powFood)
    gdis = []
    i = 1
    for g in newGhostStates:
        gpos = successorGameState.getGhostPosition(i)
        xg,yg = gpos
        a = 0.75
        if g.scaredTimer>1:
           a = a*-500
        gdis.append(a*manhattanDistance(gpos,pos))
        i += 1
    tot = sum(gdis) + 100*tot + -100*fo + -10000*powow 
    return tot

# Abbreviation

better = betterEvaluationFunction
