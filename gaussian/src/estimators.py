import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

from functools import partial

from src.utils import *

import random

def nwj_lower_bound_obj(scores):
    return tuba_lower_bound(scores - 1.)

def mine_lower_bound(f, buffer=None, momentum=0.9):
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update

def chi_lower_bound_obj(f, alpha, beta, gamma):
    f_diag = f.diag()
    first_term = (f_diag - 0.5 * beta * (f_diag ** 2)).mean()
    n = f.size(0)
    f_offdiag = f.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
    #     f_offdiag = f.masked_fill_(torch.eye(n, n).byte().cuda(), 0)
    second_term = (alpha * f_offdiag + 0.5 * gamma * (f_offdiag ** 2)).mean()
    return first_term - second_term
    
def dv_upper_lower_bound_obj(f):
    """DV lower bound, but upper bounded by using log outside."""
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term

def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    batch_size = scores.size(0)

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term

def infonce_lower_bound_obj(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi

def log_density_ratio_mi(f):
    return torch.log(torch.clamp(f, min=1e-4)).diag().mean()
   
def direct_log_density_ratio_mi(f):
    return f.diag().mean()   
    
def dv_clip_upper_lower_bound(f, alpha=1.0, clip=None):
    z = renorm_q(f, alpha, clip)
    dv_clip = f.diag().mean() - z

    return dv_clip

def js_fgan_lower_bound_obj(f):
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

    
def log_density_ratio_mi_chi(f, alpha, beta, gamma):
    f_diag = f.diag().mean()
    true_ratio = (f_diag * gamma + alpha) / (1. - f_diag * beta)
    return torch.log(torch.clamp(true_ratio, min=1e-4))

def MI_Estimator(f, train_type='nwj_lower_bound_obj', eval_type='nwj_lower_bound_obj',
                                  **kwargs):   
    if train_type == 'tuba_lower_bound' or train_type == 'mine_lower_bound'\
       or train_type == 'chi_lower_bound_obj':
        if train_type != 'chi_lower_bound_obj': 
            assert train_type == eval_type
        train_val = getattr(sys.modules[__name__], train_type)(f, **kwargs)
    else:
        train_val = getattr(sys.modules[__name__], train_type)(f)
    if train_type == eval_type:
        return train_val, train_val
    
    if train_type == 'nwj_lower_bound_obj' and eval_type == 'direct_log_density_ratio_mi':
        eval_val = getattr(sys.modules[__name__], eval_type)(f-1.) 
    elif eval_type == 'tuba_lower_bound' or eval_type == 'dv_clip_upper_lower_bound'\
         or eval_type == 'mine_lower_bound' or eval_type == 'log_density_ratio_mi_chi':
        eval_val = getattr(sys.modules[__name__], eval_type)(f, **kwargs)
    # note that especially when we use JS to train, and use nwj to evaluate
    elif eval_type == 'nwj_lower_bound_obj':
        eval_val = getattr(sys.modules[__name__], eval_type)(f+1., **kwargs)
    else:
        eval_val = getattr(sys.modules[__name__], eval_type)(f)
    
    with torch.no_grad():
        eval_train = eval_val - train_val
        
    return train_val + eval_train, train_val

def nwj_lower_bound(f):
    return MI_Estimator(f, train_type='nwj_lower_bound_obj', eval_type='nwj_lower_bound_obj')

def infonce_lower_bound(f):
    return MI_Estimator(f, train_type='infonce_lower_bound_obj', eval_type='infonce_lower_bound_obj')

def js_lower_bound(f):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', eval_type='nwj_lower_bound_obj')

def dv_upper_lower_bound(f):
    return MI_Estimator(f, train_type='dv_upper_lower_bound_obj', eval_type='dv_upper_lower_bound_obj')

def smile_lower_bound(f, alpha=1.0, clip=5.0):
    return MI_Estimator(f, train_type='js_fgan_lower_bound_obj', 
                        eval_type='dv_clip_upper_lower_bound', alpha=alpha, clip=clip)

def chi_lower_bound(f, alpha=0.01, beta = 0.005, gamma = 0.995):
    return MI_Estimator(f, train_type='chi_lower_bound_obj', eval_type='log_density_ratio_mi_chi', alpha=alpha, beta=beta, gamma=gamma)