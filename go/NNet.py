import os
import shutil
import time
from utils import dotdict
from go.go_nn import GoNNet
from NeuralNet import NeuralNet
import torch
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from pytorch_classification.utils import Bar, AverageMeter
import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,  #Originally 10
    'batch_size': 512, #Originally 64
    'cuda': torch.cuda.is_available(),
    'num_filters': 256,
    'num_blocks': 5
})

class NNetWrapper(NeuralNet):
    def __init__(self,game):
        self.nnet = GoNNet(board_size = game.n, history = game.nn_hist_len, n_blocks = args.num_blocks, n_filters = args.num_filters)
        #Not in Othello's NNetWrapper so I commented out for now
        if args.cuda:
           print("Moving model to GPU")
           self.nnet.cuda()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game
        self.history = game.nn_hist_len

    def loss_v(self,outputs,targets):
        #print(outputs.size())
        return torch.sum((outputs-targets)**2)/targets.size()[0]
    def loss_pi(self,outputs,targets):
        #print(outputs)
        #print(torch.log(outputs))
        return -torch.sum(targets*torch.log(outputs))/targets.size()[0]
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch+1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples)/args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples)/args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.Tensor(np.array(boards).astype(np.float64))
                target_pis = torch.Tensor(np.array(pis))
                target_vs = torch.Tensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                #print(self.nnet(boards))
                #print(boards.shape)
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(out_pi, target_pis)
                l_v = self.loss_v(out_v, target_vs)
                total_loss = l_pi + l_v
                #print(out_pi,out_v)

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=int(len(examples)/args.batch_size),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()


    def predict(self, board: np.ndarray):
        """
        board: np array with board
        """
        # timing
        #start = time.time()

        # preparing input
        board = torch.as_tensor(board, dtype=torch.float)
        if args.cuda:
            board = board.contiguous().cuda()
            self.nnet.cuda()
        
        board = board.view(1,2*self.history+1, self.board_x, self.board_y)
        self.nnet.eval()

        with torch.no_grad():
            p, v = self.nnet(board)

        #print(p.shape, v.shape)
        #print(v.data.cpu().numpy()[0])
        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return p.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


    