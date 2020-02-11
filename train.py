#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:59:08 2020

@author: molano
"""
import gym
import neurogym
import plotting
from neurogym.wrappers import monitor
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C  # ACER, PPO2
task = 'CVLearning-v0'

KWARGS = {'dt': 100,
          'timing': {'fixation': ('constant', 200),
                     'stimulus': ('constant', 500),
                     'delay': ('choice', [100, 300, 500]),
                     'decision': ('constant', 300)}}

th_mat = [0.3, 0.5, 0.7, 0.9]
for th in th_mat:
    KWARGS['th'] = th
    env = gym.make(task, **KWARGS)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    save_folder = '/home/molano/CV-Learning/results/train_' + str(th) + '/'
    env = monitor.Monitor(env, folder=save_folder, sv_per=10000, sv_fig=True,
                          verbose=True)
    env = DummyVecEnv([lambda: env])
    model = A2C(LstmPolicy, env, verbose=0,
                policy_kwargs={'feature_extraction': "mlp"})

    model.learn(total_timesteps=1000000)
    plotting.plot_rew_across_training(folder=save_folder)

KWARGS['init_ph'] = 4
env = gym.make(task, **KWARGS)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
save_folder = '/home/molano/CV-Learning/results/train_full/'
env = monitor.Monitor(env, folder=save_folder, sv_per=10000, sv_fig=True,
                      verbose=True)
env = DummyVecEnv([lambda: env])
model = A2C(LstmPolicy, env, verbose=0,
            policy_kwargs={'feature_extraction': "mlp"})

model.learn(total_timesteps=1000000)
plotting.plot_rew_across_training(folder=save_folder)
