#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:21:21 2020

@author: manuel
"""
from plotting import fig_
stage = 1

if stage == 0:
    start = 10
    end = 28
    dash = 2
    show_delays = False
    show_perf = False
    show_gt = False
    path = '/Users/martafradera/CV-figures/data_fig/stage_0.npz'
    folder = '/Users/martafradera/CV-figures/figures/stage_0'
elif stage == 1:
    start = 45
    end = 65
    dash = None
    show_delays = False
    show_perf = True
    show_gt = True
    path = '/Users/martafradera/CV-figures/data_fig/stage_1.npz'
    folder = '/Users/martafradera/CV-figures/figures/stage_1'
elif stage == 2:
    start = 18
    end = 40
    dash = None
    show_delays = False
    show_perf = True
    show_gt = True
    path = '/Users/martafradera/CV-figures/data_fig/stage_2.npz'
    folder = '/Users/martafradera/CV-figures/figures/stage_2'
elif stage == 3:
    start = 0
    end = 38
    dash = None
    show_delays = True
    show_perf = True
    show_gt = True
    path = '/Users/martafradera/CV-figures/data_fig/stage_3.npz'
    folder = '/Users/martafradera/CV-figures/figures/stage_3'
elif stage == 4:
    start = 45
    end = 85
    dash = None
    show_delays = True
    show_perf = True
    show_gt = True
    path = '/Users/martafradera/CV-figures/data_fig/stage_4.npz'
    folder = '/Users/martafradera/CV-figures/figures/stage_4'

fig_(path=path, obs_traces=['Fixation Cue', 'Left Stim', 'Right Stim'],
     start=start, end=end, dash=dash, show_delays=show_delays,
     show_perf=show_perf, show_gt=show_gt, folder=folder+'.svg')
