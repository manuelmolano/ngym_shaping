#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 19:12:15 2021

@author: manuel
see compute_mean_perf in ngym_priors
"""


import ngym_shaping as ngym
import numpy as np
from ngym_shaping.core import TrialWrapper
import warnings
from collections import deque


class MeanPerf(TrialWrapper):
    metadata = {
        'description': 'Computes mean for of quantity in info[key] .',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, perf_w=100, key='real_performance'):
        """
        block_nch: duration of each block containing a specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.max_nch = len(self.unwrapped.choices)  # Max number of choices
        self.perf_w = perf_w
        self.perf = deque(maxlen=perf_w)
        self.key = key

    def new_trial(self, **kwargs):
        return self.env.new_trial(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['new_trial']:
            self.perf.append(1*info[self.key])
            info['mean_perf'] = np.mean(self.perf)\
                if len(self.perf) == self.perf_w else 0
        return obs, reward, done, info
