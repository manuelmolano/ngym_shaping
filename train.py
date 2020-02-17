#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:59:08 2020

@author: molano
"""
import gym
import os
import neurogym
import plotting
from neurogym.wrappers import monitor
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, PPO2, ACKTR
num_instances = 3
main_folder = '/home/molano/CV-Learning/results_1402/'
task = 'CVLearning-v0'

KWARGS = {'dt': 100,
          'timing': {'fixation': ('constant', 200),
                     'stimulus': ('constant', 500),
                     'delay': ('choice', [100, 300, 500]),
                     'decision': ('constant', 300)}}
algs = [A2C, ACER, PPO2]
algs_names = ['A2C', 'PPO2', 'ACER']  # 'ACKTR',
th_mat = [-1, 0.3, 0.5, 0.7, 0.9]
for ind_inst in range(num_instances):
    for ind_alg, alg in enumerate(algs):
        for th in th_mat:
            if th == -1:
                save_folder =\
                    main_folder + '/train_full_' + str(ind_inst) + '_' +\
                    algs_names[ind_alg] + '/'
            else:
                save_folder =\
                    main_folder + 'train_' + str(th) + '_' + str(ind_inst) +\
                    '_' + algs_names[ind_alg] + '/'
            print(save_folder)
            if not os.path.exists(save_folder + 'bhvr_data_all.npz'):
                KWARGS['th'] = th
                env = gym.make(task, **KWARGS)
                env = monitor.Monitor(env, folder=save_folder, sv_per=10000,
                                      sv_fig=True, verbose=True)
                env = DummyVecEnv([lambda: env])
                model = alg(LstmPolicy, env, verbose=0,
                            policy_kwargs={'feature_extraction': "mlp"})
                model.learn(total_timesteps=5000000)
                plotting.plot_rew_across_training(folder=save_folder)
            else:
                print('DONE')
