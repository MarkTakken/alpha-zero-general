from Coach import Coach
from go.GoGame import GoGame as Game
from go.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 3,  #Originally 1000
    'numEps': 25,        #Originally 100    # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #Originally 15
    'updateThreshold': 0.5,     #Originally 0.6 # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    #Originally 200000 # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          #Originally 25 # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         #Originally 40 # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './trained_models/go',
    'load_model': False,
    'load_folder_file': ('./trained_models/go','best.pth.tar'), #Originally /dev/models/8x100x50 , best.pth.tar
    'numItersForTrainExamplesHistory': 50, #Originally 20

})

if __name__ == "__main__":
    g = Game(19)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
