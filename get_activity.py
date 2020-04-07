#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:33:28 2020

@author: martafradera
"""

import gym
import neurogym
import os
import numpy as np
from stable_baselines import A2C, ACER, PPO2, ACKTR
from stable_baselines.common.policies import LstmPolicy


def get_activity(env, folder, alg):

    activity_matrix = np.empty(0)

    for file in os.listdir(folder):
        print('file', file)
        #model = alg(LstmPolicy, env, verbose=0,
        #            policy_kwargs={'feature_extraction': "mlp"})
        print(folder+'/'+file)
        alg.load(folder+'/'+file)
        erojnfe
        obs = env.reset()

        states_matrix = np.empty()
        for i in range(1000):
            action, _states = model.predict(obs)   # neurons for each stage
            obs, rewards, dones, info = env.step(action)
            env.render()

            states_matrix = np.concatenate(states_matrix, _states, axis=0)

        env.close()
        states_matrix.flatten()
        np.concatenate((activity_matrix, states_matrix), axis=0)


task = 'CVLearning-v0'
KWARGS = {'dt': 100,
          'timing': {'fixation': ('constant', 200),
                     'stimulus': ('constant', 500),
                     'delay': ('choice', [0, 1000, 3000]),
                     'decision': ('constant', 300)}}
KWARGS['stages'] = [4]
env = gym.make(task, **KWARGS)
folder = '/Users/martafradera/Desktop/OneDrive - Universitat de Barcelona/TFG/models'
alg=A2C

get_activity(env, folder, alg)

