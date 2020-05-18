#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:33:28 2020

@author: martafradera
"""

import glob
import gym
import neurogym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines import A2C  # , ACER, PPO2, ACKTR
from stable_baselines.common.vec_env import DummyVecEnv
from neurogym.utils import plotting


def create_env(task, n_ch, seed):
    # task
    task_kwargs = {'dt': 100, 'th_stage': 0.75, 'keep_days': 0, 'stages': [4],
                   'n_ch': n_ch, 'timing': {'fixation': ('constant', 300),
                                            'stimulus': ('constant', 500),
                                            'delay': ('choice', [0, 1000,
                                                                 3000]),
                                            'decision': ('constant', 300)}}
    env_id = task
    env = gym.make(env_id, **task_kwargs)
    env.seed(seed)
    env = DummyVecEnv([lambda: env])
    return env


def model_fig(file, folder, protocol, n_ch, data, states):
    fname = file[file.find(protocol+'_')+len(protocol)+1:file.find('.zip')]
    fname = folder+protocol+'_n_ch_'+str(n_ch)+'_model_'+fname
    perf = np.array(data['perf'])
    perf = perf[np.where(perf != -1)]
    name = 'protocol: '+protocol +' perf: '+str(round(np.mean(perf), 2))
    plotting.fig_(data['ob'], data['actions'], gt=data['gt'],
                  rewards=data['rewards'],  states=states,
                  fname=fname, name=name)


def act_fig(activity_mat, folder, protocol, n_ch, **fig_kwargs):
    f, axes = plt.subplots(1, 1, **fig_kwargs)
    axes.imshow(activity_mat, aspect='auto', origin='lower')
    f.savefig(folder+protocol+'_n_ch_'+str(n_ch)+'_activity.png')
    plt.close(f)


def evaluate(env, model, num_steps):
    obs = env.reset()
    states = None
    done = [False]
    ob_mat = []
    act_mat = []
    rew_mat = []
    states_mat = []
    gt_mat = []
    perf = []
    for i in range(num_steps):
        ob_mat.append(obs[0])
        action, states = model.predict(obs, states, mask=done)
        act_mat.append(action)
        states_mat.append(states)
        obs, rewards, done, info = env.step(action)
        rew_mat.append(rewards)
        if len(states) > 0:
            states_mat.append(states)
        info = info[0]
        if 'gt' in info.keys():
            gt_mat.append(info['gt'])
        else:
            gt_mat.append(0)
        if info['new_trial']:
            perf.append(info['performance'])
        env.render()
    states = np.array(states_mat)
    states = states[:, 0, :]
    data = {'ob': ob_mat, 'actions': act_mat, 'gt': gt_mat,
            'rewards': rew_mat, 'states': states, 'perf': perf}
    return data


def get_activity(folder, alg, protocols, n_ch=2, seed=0, num_steps=1000,
                 sv_model=False, sv_act=True):
    for protocol in protocols:
        files = glob.glob(folder+'model_n_ch_'+str(n_ch)+'_'+protocol+'_*')
        activity_mat = None
        for file in files:
            print(file)
            model = alg.load(file)
            env = create_env('CVLearning-v0', n_ch, seed)
            # data = plotting.run_env(env, model=model, num_steps=num_steps)
            data = evaluate(env, model, num_steps)
            # z-score
            states_ori = data['states']
            states = (states_ori - np.mean(states_ori))/np.std(states_ori)
            # activity mat
            states_flat = np.transpose(states)
            states_flat = states_flat.flatten()
            if activity_mat is None:
                activity_mat = states_flat
            else:
                activity_mat = np.vstack((activity_mat, states_flat))
            # plotting
            if sv_model:
                model_fig(file, folder, protocol, n_ch, data, states)
        if sv_act:
            act_fig(activity_mat, folder, protocol, n_ch)


if __name__ == "__main__":
    n_ch = [2, 10, 20]
    alg = A2C
    protocols = ['01234', '4']
    for n in n_ch:
        folder = '/Users/martafradera/Desktop/models/'+str(n)+'_ch/'

        get_activity(folder, alg, protocols, n_ch=n, num_steps=1000, sv_model=True)
    # create_env('CVLearning-v0', 2)
