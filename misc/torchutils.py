
import torch

from torch.utils.data import Subset
from misc import google_cloud
import numpy as np
import math
import shutil
import os


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1

class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step))/2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):

    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

def load_checkpoint(args,model, optimizer):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    filename=os.path.join(args.checkpoint_dir,args.checkpoint_filename)
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #losslogger = checkpoint['losslogger']
        
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer,start_epoch,loss

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savepath = os.path.join(args.checkpoint_dir, filename)
    print("=> saving checkpoint '{}'".format(savepath))
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))
     #save to google Bucket
    google_cloud.upload_blob('hpa_1',save_path,'irn/checkpoints/')
    
