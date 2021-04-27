import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils import *
from src.estimators import *
from src.models import *

def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None, clamping_values=None, **kwargs):
    x, y = x.cuda(), y.cuda()
    scores = critic_fn(x, y)
    if clamping_values is not None:
        scores = torch.clamp(scores, min=clamping_values[0], max=clamping_values[1])
    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'nwj':
        return nwj_lower_bound(scores)
    elif estimator == 'infonce':
        return infonce_lower_bound(scores)
    elif estimator == 'js':
        return js_lower_bound(scores)
    elif estimator == 'dv':
        return dv_upper_lower_bound(scores)
    elif estimator == 'smile':
        return smile_lower_bound(scores, **kwargs)
    elif estimator == 'gen-chi':
        return chi_lower_bound(scores, **kwargs)


def train_estimator(critic_params, data_params, mi_params, opt_params, **kwargs):
    
    CRITICS = {
        'separable': SeparableCritic,
        'concat': ConcatCritic,
    }
    
    BASELINES = {
        'constant': lambda: None,
        'unnormalized': lambda: mlp(dim=data_params['dim'], \
                        hidden_dim=512, output_dim=1, layers=2, activation='relu').cuda(),
    }

    critic = CRITICS[mi_params.get('critic', 'separable')](
        rho=None, **critic_params).cuda()
    baseline = BASELINES[mi_params.get('baseline', 'constant')]()
    
    opt_crit = optim.Adam(critic.parameters(), lr=opt_params['learning_rate'])
    if isinstance(baseline, nn.Module):
        opt_base = optim.Adam(baseline.parameters(),
                              lr=opt_params['learning_rate'])
    else:
        opt_base = None

    def train_step(rho, data_params, mi_params):
        opt_crit.zero_grad()
        if isinstance(baseline, nn.Module):
            opt_base.zero_grad()

        if mi_params['critic'] == 'conditional':
            critic_ = CRITICS['conditional'](rho=rho).cuda()
        else:
            critic_ = critic

        x, y = sample_correlated_gaussian(
            dim=data_params['dim'], rho=rho,\
                batch_size=data_params['batch_size'], cubic=data_params['cubic'])
        if False:
            mi, p_norm = estimate_mutual_information(
                mi_params['estimator'], x, y, critic_, baseline,\
                    mi_params.get('alpha_logit', None), **kwargs)
        else:
            mi, train_obj = estimate_mutual_information(
                mi_params['estimator'], x, y, critic_, baseline,\
                    mi_params.get('alpha_logit', None), **kwargs)
        loss = -mi

        loss.backward()
        opt_crit.step()
        if isinstance(baseline, nn.Module):
            opt_base.step()
        
        if False:
            return mi, p_norm
        else:
            return mi, train_obj

    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

    if False:
        estimates = []
        p_norms = []
        for i in range(opt_params['iterations']):
            mi, p_norm = train_step(
                rhos[i], data_params, mi_params)
            mi = mi.detach().cpu().numpy()
            p_norm = p_norm.detach().cpu().numpy()
            estimates.append(mi)
            p_norms.append(p_norm)
        
        return np.array(estimates), np.array(p_norms)
    else:
        estimates = []
        train_objs = []
        for i in range(opt_params['iterations']):
            mi, train_obj = train_step(
                rhos[i], data_params, mi_params)
            mi = mi.detach().cpu().numpy()
            train_obj = train_obj.detach().cpu().numpy()
            estimates.append(mi)
            train_objs.append(train_obj)

        return np.array(estimates), np.array(train_objs)
