#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:33:28 2020

@author: martafradera
"""

import os
import gym
import neurogym
import importlib
import numpy as np
from stable_baselines import A2C  # , ACER, PPO2, ACKTR
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from neurogym.wrappers import ALL_WRAPPERS


def create_env(task, n_ch, seed):
    n_cpu = 1

    # task
    task_kwargs = {'dt': 100, 'th_stage': 0.75, 'keep_days': 0, 'stages': [4],
                   'n_ch': n_ch, 'timing': {'fixation': ('constant', 300),
                                            'stimulus': ('constant', 500),
                                            'delay': ('choice', [0, 1000, 3000]),
                                            'decision': ('constant', 300)}}
    # wrappers
    # wrappers_kwargs = {'Monitor-v0': {'folder': '', 'sv_fig': False,
    #                                  'sv_per': 10000, 'fig_type': 'svg'}}

    env_id = task
    env = gym.make(env_id, **task_kwargs)
    env.seed(seed)
    env = DummyVecEnv([lambda: env])
    # env = SubprocVecEnv([env for i in range(n_cpu)])  # ???
    return env


def load_model(folder, alg):
    file = folder+'model.zip'
    model = alg.load(file)
    return model


def get_activity(folder, alg, n_ch, seed):
    activity_matrix = np.empty(0)
    print('creating env')
    for file in os.listdir(folder):
        print(file)
        print('loading model')
        model = alg.load(folder+file)
        print('getting activity')
        env = create_env('CVLearning-v0', n_ch, seed)
        obs = env.reset()
        total_obs = obs
        states_matrix = np.empty(0)
        for i in range(50):
                action, _states = model.predict(obs)   # neurons for each stage
                obs, rewards, dones, info = env.step(action)
                env.render()
                states_matrix = np.concatenate((states_matrix, _states[0]),
                                               axis=0)
                total_obs = np.concatenate((total_obs, obs))
        env.close()
        states_matrix.flatten()
        activity_matrix = np.concatenate((activity_matrix, states_matrix),
                                         axis=0)


if __name__ == "__main__":
    folder = '/Users/martafradera/Desktop/models/'
    alg = A2C
    n_ch = 2
    seed = 0

    get_activity(folder, alg, n_ch, seed)
    # create_env('CVLearning-v0', 2)
