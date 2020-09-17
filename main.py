import logging
from sys import argv

from Coach import Coach
from go.GoGame import GoGame as Game
from go.NNet import NNetWrapper as nn
#In order to run for Othello, comment the above two lines and uncomment the below two lines
#from othello.OthelloGame import OthelloGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn
from utils import *
#Random edit
logging.basicConfig(
    format='%(levelname)-8s [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
root_logger = logging.getLogger()

if "-d" in argv or "--debug" in argv:
    root_logger.setLevel(logging.DEBUG)

args = dotdict({
    'startIter': 2,
    'numIters': 1000,  #Originally 1000
    'numEps': 60,        #Originally 100    # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 9,        #Originally 15
    'updateThreshold': 0.5,     #Originally 0.6 # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    #Originally 200000 # Number of game examples to train the neural networks.
    'numMCTSSims': 40,          #Originally 25 # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         #Originally 40 # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './trained_models/9x9go',
    'load_model': True,
    'load_folder_file': ('./trained_models/9x9go','checkpoint_1.pth.tar'), #Originally /dev/models/8x100x50 , best.pth.tar
    'load_folder_file_examples': ('./trained_models/9x9go','checkpoint_0.pth.tar.examples'), #added
    'skipFirstSelfPlay': False,
    'numItersForTrainExamplesHistory': 100, #Originally 20
    'resignationThreshold': -0.90  #Added
})

if __name__ == "__main__":
    g = Game(9)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
