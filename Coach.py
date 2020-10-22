from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import logging

import tqdm
import itertools
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from typing import List, Tuple
logger = logging.getLogger(__file__)
import platform


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, pnet, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = pnet #self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        #self.skipFirstSelfPlay = False    # can be overriden in loadTrainExamples()

    @staticmethod
    def executeEpisode(worker_index: int,
                       game,
                       args,
                       nnet):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        # Input arguments:
        # game = self.game
        # args = self.args
        # nnet = self.nnet
        mcts = MCTS(game, nnet, args)
        #
        trainExamples = []
        board = game.getInitBoard()
        curPlayer = 1
        episodeStep = 0
        #prev_state_action = None
        
        # Create a neat progress bar for each worker.
        pbar = tqdm.tqdm(
            itertools.count(),
            desc=f"Worker {worker_index}, Episode step",
            position=worker_index + 1, # + 1 so we keep the "Self Play using ..." progress bar on top.
            leave=False,
            # NOTE: Set this to True to disable the pbars for each worker.
            disable=False,
        )
        
        for episodeStep in pbar:
            canonicalBoard = game.getCanonicalForm(board, curPlayer)
            #v = self.nnet.predict(canonicalBoard)[1]
            #print(v)
            #if v < self.args.resignationThreshold:
            #    return [(x[0],x[2],-1*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples] 
            temp = int(episodeStep < args.tempThreshold)
            pi, root_q = mcts.getActionProbAndRootValue(board, curPlayer, temp=temp)
            #logger.info(root_q)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, curPlayer, p, None])

            if args.resignationOn and root_q < args.resignationThreshold:
                logger.info((episodeStep,root_q,board))
                #Player resigns (I added this)
                break

            action = np.random.choice(len(pi), p=pi)
            prev_state_action = game.stringRepresentation(board),action
            board, curPlayer = game.getNextState(board, curPlayer, action)

            r = game.getGameEnded(board, curPlayer)
            #print(board)
            if r!=0:
                #logger.info(episodeStep)
                break
            
            # Display some information in the pbar, just for fun.
            pbar.set_postfix(root_q=root_q)
            
            # TODO: Remove this, I'm just using this to debug the mp stuff below.
            # if episodeStep > 100:
            #     break
        
        return [
            (x[0], x[2], -1*((-1) ** (x[1]!=curPlayer)))
            for x in trainExamples
        ]
        
        
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(self.args.startIter, self.args.numIters+self.args.startIter):
            # bookkeeping
            print('------ITER ' + str(i) + '------')
            # examples of the iteration
            if not self.args.skipFirstSelfPlay or i > self.args.startIter: #or i > 1
                
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                n_processes: int = 4
                # or, if you have enough compute (and VRAM):
                # n_processes = mp.cpu_count()
                
                with mp.Pool(n_processes) as pool:
                    self.nnet.nnet.share_memory()
                    pbar = tqdm.tqdm(range(self.args.numEps), position=0)
                    pbar.set_description(f"Self Play using {n_processes} processes")

                    for eps in pbar:
                        # Arguments for each worker.
                        worker_args = [
                            (i, self.game, self.args, self.nnet)
                            for i in range(n_processes)
                        ]
                        # Apply the executeEpisode method on each argument:
                        for worker_examples in pool.starmap(Coach.executeEpisode, worker_args):
                            iterationTrainExamples.extend(worker_examples)

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
  
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i-1)
            
            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            if not self.args.skipFirstTrain or i > self.args.startIter:
                # training new network, keeping a copy of the old one
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='old.pth.tar')
                self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='old.pth.tar')
                #pmcts = MCTS(self.game, self.pnet, self.args)
            
                self.nnet.train(trainExamples)
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='new.pth.tar')
                #nmcts = MCTS(self.game, self.nnet, self.args)

            pmcts = MCTS(self.game, self.pnet, self.args)
            nmcts = MCTS(self.game, self.nnet, self.args)
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(pmcts,nmcts,self.game,self.args.resignationOn,self.args.resignationThreshold)
            #arena = Arena(lambda state,player: np.where(pmcts.getActionProb(state,player, temp=0) == 1)[0][0],
            #              lambda state,player: np.where(nmcts.getActionProb(state,player, temp=0) == 1)[0][0], self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='old.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        #modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        #examplesFile = modelFile+".examples"
        examplesFile = os.path.join(self.args.load_folder_file_examples[0], self.args.load_folder_file_examples[1])
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            #self.skipFirstSelfPlay = True
