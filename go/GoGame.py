import numpy as np 
from Game import Game
from copy import deepcopy
import logging
logger = logging.getLogger(__file__)

class GoGame(Game):
    square_content = {
        -1: "O",
        +0: "-",
        +1: "X"
    }


    def __init__(self,n=9,nn_hist_len=2,komi=7.5):
        self.n = n
        self.nn_hist_len = nn_hist_len
        self.komi = komi #Compensation given to white for going second

    #The structure of the state must include history as well as the current board in Go.
    def getInitBoard(self):
        return np.zeros((1,self.n,self.n),dtype=int)

    def getBoardSize(self):
        return (self.n,self.n)

    #n*n spots to place a stone, plus 1 for pass
    def getActionSize(self):
        return self.n * self.n + 1

    #This is the only function I would have to change for Toroidal Go.
    def getSurroundingPoints(self,board,point):
        points = []
        for direction in [(1,0),(0,1),(-1,0),(0,-1)]:
            new_point = (point[0]+direction[0],point[1]+direction[1])
            if 0 <= new_point[0] <= self.n - 1 and 0 <= new_point[1] <= self.n - 1: #Checks whether newPoint is off the board, which is not allowed
                points.append(new_point)
        return points 
    
    #Recursively obtains the group assossiated with a point and the liberties of the group (vacant spots it touches)
    def getGroupAndLiberties(self,board,start_point,start_group=[],start_liberties=[]):
        color = board[start_point[0],start_point[1]]
        group = start_group + [start_point]
        liberties = deepcopy(start_liberties)
        for point in self.getSurroundingPoints(board,start_point):
            content = board[point[0]][point[1]]
            if content == 0 and not(point in liberties):
                liberties.append(point)
            elif content == color and not(point in group):
                (group,liberties) = self.getGroupAndLiberties(board,point,group,liberties)
        return (group,liberties)

    def removeGivenStones(self,board,points):
        board = deepcopy(board)
        for point in points:
            board[point[0]][point[1]] = 0
        return board

    #Removes any groups of color that have no liberties
    #Returns board and whether anything was captured
    def clearStones(self,board,color,last_move):
        board = deepcopy(board)
        captured_enemy = False
        checked = [] 
        for (r,c) in self.getSurroundingPoints(board,last_move):
            if board[r][c] == color and not((r,c) in checked):
                (group,liberties) = self.getGroupAndLiberties(board,(r,c))
                if len(liberties) == 0:
                    captured_enemy = True
                    board = self.removeGivenStones(board,group)
                else:
                    checked += group
        return board, captured_enemy

    #Move is in the form (x,y)
    #Returns the next board, whether any enemy groups were captured and whether the player committed suicide
    def getNextBoard(self,board,player,move):
        board = deepcopy(board)
        suicide = False
        captured_enemy = False
        if move != "pass":
            board[move[0]][move[1]] = player
            board,captured_enemy = self.clearStones(board,-player,move)
            if not(captured_enemy):
                (_,liberties) = self.getGroupAndLiberties(board,move)
                if len(liberties) == 0:
                    suicide = True
        return board,captured_enemy,suicide

    #Appends new board to state, returns state
    #Action is a number from 0 to n*n
    def getNextState(self,state,player,action):
        board = state[len(state)-1]
        move = "pass" if action == self.n*self.n else (action // self.n, action % self.n)
        newboard = np.array([self.getNextBoard(board,player,move)[0]])
        return np.append(state,newboard,0),-player

    #Move is in the form (x,y)
    def isValidMove(self,state,player,move):
        board = state[len(state)-1]
        if board[move[0]][move[1]] != 0: #Spot must be empty in order to be valid
            return False
        newboard,captured_enemy,suicide = self.getNextBoard(board,player,move)
        if suicide:
            return False #Suicide is illegal
        if not(captured_enemy): #Don't need to check whether you're repeating a situation
            return True 
        i = len(state)-2
        while i >= 0:
            if np.array_equal(newboard,state[i]):
                return False  #Repeating situations is illegal
            i -= 2
        return True

    #Returns a binary vector of length n*n+1 where index i has a 1 if the move is valid and otherwise has a 0
    def getValidMoves(self,state,player):
        actions = [0]*self.getActionSize()
        for action in range(self.n*self.n):
            move = (action // self.n, action % self.n)
            if self.isValidMove(state,player,move):
                actions[action] = 1
        actions[-1] = 1
        return actions

    #Recursively obtains the territory associated with the point and the border of the territory and the content of the border
    #Same idea as in getGroupAndLiberties
    def getTerritoryAndBorder(self,board,start_point,start_territory=[],start_border=[],start_border_content=[]):
        territory = start_territory + [start_point]
        border = deepcopy(start_border)
        border_content = deepcopy(start_border_content)
        for point in self.getSurroundingPoints(board,start_point):
            content = board[point[0]][point[1]]
            if content == 0 and not(point in territory):
                (territory,border,border_content) = self.getTerritoryAndBorder(board,point,territory,border,border_content)
            elif (content == 1 or content == -1) and not(point in border):
                border.append(point)
                border_content.append(content)
        return (territory,border,border_content) 

    #Scores the terminal board and determines the winner
    def getWinner(self,terminal_board):
        black_score = 0
        white_score = self.komi
        checked = []
        for r in range(self.n):
            for c in range(self.n):
                if (r,c) in checked:
                    continue
                content = terminal_board[r][c]
                if content == 1:
                    black_score += 1
                elif content == -1:
                    white_score += 1
                else:
                    (territory,_,border_content) = self.getTerritoryAndBorder(terminal_board,(r,c))
                    checked += territory
                    if len(border_content) > 0:
                        border_content = np.array(border_content)
                        if all(border_content == 1):
                            black_score += len(territory)
                        elif all(border_content == -1):
                            white_score += len(territory)
        return (1 if black_score > white_score else -1)

    def isTerminal(self,state):
        length = len(state)
        return (len(state) >= 3 and np.array_equal(state[length-1],state[length-2]) and np.array_equal(state[length-2],state[length-3])) or len(state) > 2 * self.n * self.n

    #Returns reward for player
    def getGameEnded(self,state,player):
        return player * self.getWinner(state[len(state)-1]) if self.isTerminal(state) else 0

    #Includes self.nn_hist_len boards for history
    def getCanonicalForm(self,state,player):
        length = len(state)
        if length >= self.nn_hist_len:
            trimmedHistory = player * state[length-self.nn_hist_len:]
        else:
            trimmedHistory = np.concatenate((np.zeros((self.nn_hist_len-length,self.n,self.n),dtype=int),state),axis=0)
        canonical = np.array([np.ones((self.n,self.n),dtype=int)]) if player == 1 else np.array([np.zeros((self.n,self.n),dtype=int)])
        for board in trimmedHistory:
            canonical = np.append(canonical,np.array([[[(1 if elmnt == -1 else 0) for elmnt in row] for row in board]]),axis=0)
            canonical = np.append(canonical,np.array([[[(1 if elmnt == 1 else 0) for elmnt in row] for row in board]]),axis=0)
        return canonical

    #All rotations and reflections
    def getSymmetries(self,canonicalState,pi):
        pi_board = np.reshape(pi[:-1],(self.n,self.n))
        symmetries = []
        for i in range(4):
            for j in [True,False]:
                newS = np.array([np.rot90(b,i) for b in canonicalState])
                newPi = np.rot90(pi_board,i)
                if j:
                    newS = np.array([np.fliplr(b) for b in newS])
                    newPi = np.fliplr(newPi)
                symmetries.append((newS,list(newPi.ravel())+[pi[-1]]))
        return symmetries

    def stringRepresentation(self, state: np.ndarray):
        logger.debug(f"id of state is {id(state)}")
        return state.tostring()

    @staticmethod
    def display(state):
        board = state[-1]
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(GoGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")

class Tests:

    @staticmethod
    def getInitBoard():
        g = GoGame(4,2,0)
        print(g.getInitBoard())

    @staticmethod
    def getNextStateTest():
        g = GoGame(4,2,0)
        state = np.array([[[0,-1,-1,0],[-1,1,1,-1],[-1,-1,1,1],[-1,1,1,0]]])
        print(state)
        print(g.getNextState(state,-1,15))

    @staticmethod
    def getValidMovesTest():
        g = GoGame(4,2,0)
        state = np.array([[[0,-1,1,0],[-1,1,0,1],[0,-1,1,0],[0,0,0,0]],[[0,-1,1,0],[-1,0,-1,1],[0,-1,1,0],[0,0,0,0]]])
        print(state)
        print(g.getValidMoves(state,1))

    @staticmethod
    def getValidMovesTest2():
        g = GoGame(3,2,0)
        state = np.array([[[1,0,-1],[-1,-1,-1],[0,0,0]]])
        print(state)
        print(g.getValidMoves(state,1))

    @staticmethod
    def getTerritoryAndBorderTest():
        g = GoGame(4,2,0)
        board = np.array([[0,1,0,0],[0,0,0,1],[0,1,1,0],[1,0,0,1]])
        print(board)
        print(g.getTerritoryAndBorder(board,(1,1)))

    @staticmethod
    def getTerritoryAndBorderTest2():
        g = GoGame(4,2,0)
        board = np.array([[0,-1,0,0],[1,0,0,-1],[0,-1,-1,0],[-1,0,0,-1]])
        print(board)
        print(g.getTerritoryAndBorder(board,(1,1)))

    @staticmethod
    def getWinnerTest():
        g = GoGame(7,2,0.5)
        board = np.array([[0,1,0,1,-1,0,0],[0,0,0,1,-1,0,0],[1,0,0,1,-1,-1,0],[1,1,1,-1,-1,0,-1],[0,1,1,-1,0,-1,0],[1,1,1,-1,0,-1,0],[0,1,1,-1,0,-1,0]])
        print(board)
        print(g.getWinner(board))

    @staticmethod
    def getSymmetriesTest():
        g = GoGame(4,2,0)
        canonicalState = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[17,18,19,20],[21,22,23,24],[25,26,27,28],[29,30,31,32]]])
        pi = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
        print(canonicalState)
        print(pi)
        print(g.getSymmetries(canonicalState,pi))

    @staticmethod
    def getCanonicalFormTest():
        g = GoGame(4,2,0)
        state = np.array([[[1,0,-1,1],[0,-1,0,0],[0,1,-1,0],[-1,0,0,1]],[[0,1,0,-1],[1,0,-1,-1],[1,1,1,-1],[0,0,-1,0]],[[0,0,1,0],[-1,0,0,0],[-1,-1,0,0],[1,1,1,-1]]])
        print(state)
        print('-----------')
        print(g.getCanonicalForm(state,1))

    @staticmethod
    def getCanonicalFormTest2():
        g = GoGame(4,4,0)
        state = np.array([[[1,0,-1,1],[0,-1,0,0],[0,1,-1,0],[-1,0,0,1]],[[0,1,0,-1],[1,0,-1,-1],[1,1,1,-1],[0,0,-1,0]],[[0,0,1,0],[-1,0,0,0],[-1,-1,0,0],[1,1,1,-1]]])
        print(state)
        print('-----------')
        print(g.getCanonicalForm(state,1))

    @staticmethod
    def getCanonicalFormTest3():
        g = GoGame(4,2,0)
        state = np.array([])
        print(g.getCanonicalForm(state,1))

    @staticmethod
    def isTerminalTest():
        g = GoGame(4,2,0)
        print(g.isTerminal(g.getInitBoard()))

