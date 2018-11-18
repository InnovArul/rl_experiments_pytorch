from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json, math
import os.path as osp
import torch
import numpy as np
import visdom

def mkdir_if_missing(directory):
    """to create a directory
    
    Arguments:
        directory {str} -- directory path
    """

    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

# function to freeze certain module's weight
def freeze_weights(to_be_freezed):
    for param in to_be_freezed.parameters():
        param.requires_grad = False

    for module in to_be_freezed.children():
        for param in module.parameters():
            param.requires_grad = False

def load_pretrained_model(model, pretrained_model_path):
    '''To load the pretrained model considering the number of keys and their sizes
    
    Arguments:
        model {loaded model} -- already loaded model
        pretrained_model_path {str} -- path to the pretrained model file
    
    Raises:
        IOError -- if the file path is not found
    
    Returns:
        model -- model with loaded params
    '''

    if not os.path.exists(pretrained_model_path):
        raise IOError("Can't find pretrained model: {}".format(pretrained_model_path))

    print("Loading checkpoint from '{}'".format(pretrained_model_path))
    pretrained_state = torch.load(pretrained_model_path)['state_dict']
    print(len(pretrained_state), ' keys in pretrained model')

    current_model_state = model.state_dict()
    print(len(current_model_state), ' keys in current model')
    pretrained_state = { key:val 
                        for key,val in pretrained_state.items() 
                        if key in current_model_state and val.size() == current_model_state[key].size() }

    print(len(pretrained_state), ' keys in pretrained model are available in current model')
    current_model_state.update(pretrained_state)
    model.load_state_dict(current_model_state)    
    return model

def get_currenttime_prefix():
    '''to get a prefix of current time
    
    Returns:
        [str] -- current time encoded into string
    '''

    from time import localtime, strftime
    return strftime("%d-%b-%Y_%H:%M:%S", localtime()) 

def get_learnable_params(model):
    '''to get the list of learnable params
    
    Arguments:
        model {model} -- loaded model
    
    Returns:
        list -- learnable params
    '''

    # list down the names of learnable params
    details = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            details.append((name, param.shape))
    print('learnable params (' + str(len(details)) + ') : ', details)

    #  short list the params which has requires_grad as true
    learnable_params = [param for param in model.parameters()
                              if param.requires_grad]

    print("Model size: {:.5f}M".format(sum(p.numel() for p in learnable_params)/1000000.0))
    return learnable_params    

def get_visdom_for_current_run(run_name):
    envname = get_currenttime_prefix() + '_' + run_name
    vis = visdom.Visdom(env=envname)    

    # log file name
    vis.log_to_filename = os.path.join('../scratch', envname)
    
    return vis

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, vis, title, xlabel, ylabel):
        self.reset()
        self.vis = vis
        self.value_plot = vis.line([0], opts={
            'title':title + '_current_value',
            'xlabel':xlabel,
            'ylabel':ylabel
        })

        self.avg_plot = vis.line([0], opts={
            'title':title + '_average',
            'xlabel':xlabel,
            'ylabel':ylabel
        })        

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vis.line([self.val], [self.count], win=self.value_plot, update='append')
        self.vis.line([self.avg], [self.count], win=self.avg_plot, update='append')

