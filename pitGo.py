import Arena
from MCTS import MCTS
from go.GoGame import GoGame
from go.GoPlayers import *
from go.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = GoGame()

# all players
rp = RandomPlayer(g).play
gp = GreedyGoPlayer(g).play
hp = HumanGoPlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./trained_models/go/','best.pth.tar') #./pretrained_models/tictacote/keras/, best-25eps-25sim-10epch
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda state,player: np.argmax(mcts1.getActionProb(state,player, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_checkpoint('./trained_models/go/', 'best.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)
    n2p = lambda state,player: np.argmax(mcts2.getActionProb(state,player, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=GoGame.display)

print(arena.playGames(2, verbose=True))