#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:59:08 2020

@author: molano
"""
import gym
import os
import neurogym
import plotting as pl
import matplotlib.pyplot as plt
from neurogym.wrappers import monitor
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER  # , PPO2, ACKTR
num_instances = 9
main_folder = '/Users/martafradera/CV'
task = 'CVLearning-v0'
KWARGS = {'dt': 100,
          'timing': {'fixation': ('constant', 200),
                     'stimulus': ('constant', 500),
                     'delay': ('choice', [0, 1000, 3000]),
                     'decision': ('constant', 300)}}
algs = [A2C]  # , PPO2, ACKTR]
algs_names = ['A2C']  # , 'PPO2', 'ACKTR']
th_mat = [0.7]
days_mat = [300]
for ind_inst in range(num_instances):
    for ind_alg, alg in enumerate(algs):
        for th in th_mat:
            if th == -1:
                d_mat = [0]
            else:
                d_mat = days_mat
            for d in d_mat:
                if th == -1:
                    save_folder =\
                        main_folder + '/train_full_' + str(ind_inst) + '_' +\
                        algs_names[ind_alg] + '/'
                else:
                    save_folder =\
                        main_folder + 'train_' + str(th) + '_' +\
                        str(d) + '_' + str(ind_inst) + '_' +\
                        algs_names[ind_alg] + '/'
                print(save_folder)
                if not os.path.exists(save_folder + 'bhvr_data_all.npz'):
                    KWARGS['th_stage'] = th
                    # KWARGS['perf_w_stage'] = w
                    KWARGS['trials_day'] = d
                    env = gym.make(task, **KWARGS)
                    env = monitor.Monitor(env, folder=save_folder,
                                          sv_per=1000, sv_fig=False,
                                          verbose=True, fig_type='svg')
                    env = DummyVecEnv([lambda: env])
                    model = alg(LstmPolicy, env, verbose=0,
                                policy_kwargs={'feature_extraction': "mlp"})
                    model.learn(total_timesteps=100000)
                    pl.plot_rew_across_training(folder=save_folder,
                                                conv=[1, 1, 0],
                                                metrics={'reward': [],
                                                         'performance': [],
                                                         'curr_ph': []})
                    plt.close('all')
                else:
                    print('DONE')
