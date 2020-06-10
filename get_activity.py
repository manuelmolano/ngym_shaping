#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:33:28 2020

@author: martafradera
"""
import glob
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import AgglomerativeClustering, KMeans
from stable_baselines import A2C  # , ACER, PPO2, ACKTR
from stable_baselines.common.vec_env import DummyVecEnv
from neurogym.utils import plotting


PRTCLS_IND_MAP = {'01234': 1, '4': 2}
clrs = ['c', 'm']


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


def model_fig(file, folder, protocol, n_ch, data, sv_model):
    tag = file[file.find(protocol+'_')+len(protocol)+1:file.find('.zip')]
    if sv_model:
        states_ = data['cell_states_hidden']
        # zscore
        states = np.empty(0)
        for neuron in states_:
            nact = (neuron - np.mean(neuron))/np.std(neuron)
            if len(states) == 0:
                states = nact
            else:
                states = np.vstack((states, nact[:, ]))

        states = (states - np.mean(states))/np.std(states)
        fname = folder+protocol+'_n_ch_'+str(n_ch)+'_model_'+tag
        perf = np.array(data['perf'])
        perf = perf[np.where(perf != -1)]
        name = 'protocol: ' + protocol + ' perf: ' + str(round(np.mean(perf),
                                                               2))
        plotting.fig_(data['ob'], data['actions'], gt=data['gt'],
                      rewards=data['rewards'],
                      states=states, fname=fname, name=name)
    return tag


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
    hidden_states = states[:, int(states.shape[1]/2):].T
    data = {'ob': ob_mat, 'actions': act_mat, 'gt': gt_mat,
            'hidden_states': hidden_states, 'rewards': rew_mat,
            'cell_states_hidden': states, 'perf': perf}
    return data


def activity_variance(data):
    act_ = data['hidden_states']
    # zscore
    act = np.empty(0)
    for neuron in act_:
        nact = (neuron - np.mean(neuron))/np.std(neuron)
        if len(act) == 0:
            act = nact
        else:
            act = np.vstack((act, nact[:, ]))
    obs = data['ob']
    # fixation [1., 0., 0.]
    # stimulus [1., x., x.]
    # delay [1., 0., 0.]
    # decision [0., 0., 0.]

    act_variance = []
    for neuron in range(256):
        nact = act[neuron, :]
        fixation_ind = []
        stimulus_ind = []
        delay_ind = []
        decision_ind = []
        # mean for each period per trial
        fix_means = []
        stim_means = []
        dly_means = []
        dcsn_means = []
        for ind, ob in enumerate(obs):
            if np.array_equal(ob, [1., 0., 0.]):
                if ind-1 in decision_ind:   # indicating new trial
                    fix_means.append(np.mean(nact[fixation_ind]))
                    fixation_ind.clear()
                    stim_means.append(np.mean(nact[stimulus_ind]))
                    stimulus_ind.clear()
                    if len(delay_ind) > 0:
                        dly_means.append(np.mean(nact[delay_ind]))
                        delay_ind.clear()
                    dcsn_means.append(np.mean(nact[decision_ind]))
                    decision_ind.clear()
                if ind-1 in stimulus_ind or ind-1 in delay_ind:
                    delay_ind.append(ind)
                else:
                    fixation_ind.append(ind)
            elif np.array_equal(ob, [0., 0., 0.]):
                decision_ind.append(ind)
            else:
                stimulus_ind.append(ind)

        std = [np.std(fix_means), np.std(stim_means), np.std(dly_means),
               np.std(dcsn_means)]
        if len(act_variance) == 0:
            act_variance = std
        else:
            act_variance = np.vstack((act_variance, std))
    return act_variance


def get_activity(folder, alg, protocols, n_ch=2, seed=1, num_steps=1000,
                 sv_model=False, sv_act=False):
    f0, axs0 = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    axs0 = axs0.flatten()
    for protocol in protocols:
        files = glob.glob(folder+'model_n_ch_'+str(n_ch)+'_'+protocol+'_*')
        activity_mat = None
        total_states = []
        mean_states = []
        std_states = []
        perf_th = False
        for file in files:
            print(file)
            model = alg.load(file)
            env = create_env('CVLearning-v0', n_ch, seed)
            data = evaluate(env, model, num_steps)
            states = data['hidden_states']
            # plotting model
            tag = model_fig(file, folder, protocol, n_ch, data, sv_model)
            # activity mat
            perf = np.array(data['perf'])
            perf = perf[np.where(perf != -1)]
            perf = round(np.mean(perf), 2)
            print('perf', perf)
            if perf >= 0.7:
                perf_th = True
                evaluate_connectivity(model, data, folder, protocol, tag)
                total_states.append(states)
                fr_mean, fr_std = analyze_activity(states, folder, protocol,
                                                   n_ch, tag)
                mean_states.append(np.mean(fr_mean))
                std_states.append(np.mean(fr_std))
                if activity_mat is None:
                    activity_mat = states
                else:
                    activity_mat = np.concatenate((activity_mat, states),
                                                  axis=0)
                activity_mat = np.concatenate((activity_mat,
                                               np.ones((25, num_steps))),
                                              axis=0)
        if perf_th is True:
            f, corrs_mean, corrs_std = corr_plot(total_states)
            f.savefig(folder + protocol + '_n_ch_'+str(n_ch) +
                      '_correlation.png')
            plt.close(f)
            plot_results(axs0, protocol, mean_states, std_states, corrs_mean,
                         corrs_std)
            if sv_act:
                print(np.shape(activity_mat))
                act_fig(activity_mat, folder, protocol, n_ch)
    axs0[0].set_title('Firing Rate Mean')
    axs0[0].set_xticks([1, 2])
    axs0[0].set_xticklabels(['01234', '4'])
    axs0[0].set_xlim(0, 3)
    axs0[1].set_title('Firing Rate Std')
    axs0[1].set_xticks([1, 2])
    axs0[1].set_xticklabels(['01234', '4'])
    axs0[1].set_xlim(0, 3)
    axs0[2].set_title('Correlation Mean')
    axs0[2].set_xticks([1, 2])
    axs0[2].set_xticklabels(['01234', '4'])
    axs0[2].set_xlim(0, 3)
    axs0[3].set_title('Correlation Std')
    axs0[3].set_xticks([1, 2])
    axs0[3].set_xticklabels(['01234', '4'])
    axs0[3].set_xlim(0, 3)
    f0.savefig(folder + 'firing_rate_results.png')
    plt.close(f0)


def evaluate_connectivity(model, data, folder, protocol, tag):
    params = model.get_parameters()
    wh = params['model/lstm1/wh:0']
    in_gate, forget_gate, out_gate, cell_candidate = np.split(wh, 4, axis=1)
    wh = in_gate + forget_gate + out_gate + cell_candidate
    act_variance = activity_variance(data)
    # cluster_model = AgglomerativeClustering(connectivity=wh, n_clusters=4)
    cluster_model = KMeans(n_clusters=4)  # , algorithm='full', n_init=20,
    # random_state=0)
    clustering = cluster_model.fit_predict(act_variance)
    optimal_k(act_variance, folder, protocol, tag)
    # plot_clusters(clustering, act_variance, folder, protocol, tag)
    plot_clusters2(clustering, act_variance, folder, protocol, tag)


def optimal_k(act_variance, folder, protocol, tag):
    f, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
    X = act_variance
    Nc = range(1, 20)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    ssd = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
    axs.plot(Nc, ssd)
    axs.set_xlabel('Number of Clusters')
    axs.set_ylabel('Sum of squared distance')
    f.savefig(folder+protocol+'_model_'+tag+'_k')
    plt.close(f)


def plot_clusters(clustering, act_variance, folder, protocol, tag):
    f, axs = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))
    ax = axs.flatten()
    n_clusters = np.unique(clustering)
    X = act_variance
    for cluster in n_clusters:
        row_ix = np.where(clustering == cluster)
        ax[0].scatter(X[row_ix, 0], X[row_ix, 1])
        ax[1].scatter(X[row_ix, 0], X[row_ix, 2])
        ax[2].scatter(X[row_ix, 0], X[row_ix, 3])
        ax[3].scatter(X[row_ix, 1], X[row_ix, 2])
        ax[4].scatter(X[row_ix, 1], X[row_ix, 3])
        ax[5].scatter(X[row_ix, 2], X[row_ix, 3])
    ax[0].set_xlabel('Fixation Std')
    ax[0].set_ylabel('Stimulus Std')
    ax[1].set_xlabel('Fixation Std')
    ax[1].set_ylabel('Delay Std')
    ax[2].set_xlabel('Fixation Std')
    ax[2].set_ylabel('Decision Std')
    ax[3].set_xlabel('Stimulus Std')
    ax[3].set_ylabel('Delay Std')
    ax[4].set_xlabel('Stimulus Std')
    ax[4].set_ylabel('Decision Std')
    ax[5].set_xlabel('Delay Std')
    ax[5].set_ylabel('Decision Std')

    f.savefig(folder+protocol+'_model_'+tag+'_clusters')
    plt.close(f)


def plot_clusters2(clustering, act_variance, folder, protocol, tag):
    n_clusters = np.unique(clustering)

    figsize = (4, 3)
    f = plt.figure(figsize=figsize)
    rect = [0.1, 0.1, 0.7, 0.85]
    rect_color = [0.75, 0.1, 0.08, 0.85]

    ax2 = f.add_axes(rect_color)
    # cmap = ['r', 'c', 'm', 'y']
    cmap = mpl.colors.ListedColormap(['r', 'c', 'm', 'y'])
    bounds = [0]

    ax1 = f.add_axes(rect)
    X = act_variance
    clusters_mat = np.empty(0)
    neurons = 0
    for cluster in n_clusters:
        row_ix = np.where(clustering == cluster)
        neurons += len(row_ix[0])
        bounds.append(neurons)
        if len(clusters_mat) == 0:
            clusters_mat = X[row_ix, :][0]
        else:
            clusters_mat = np.vstack((clusters_mat, X[row_ix, :][0]))
    ax1.imshow(clusters_mat, aspect='auto', origin='lower')
    labels = ['Fixation', 'Stimulus', 'Delay', 'Decision']
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Neurons')

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    spacing='proportional',
                                    orientation='vertical')
    cb2.set_label('Clusters')
    f.savefig(folder+protocol+'_model_'+tag+'_clusters_std')
    plt.close(f)


def analyze_activity(states, folder, protocol, n_ch, tag):
    fr_mean = []
    fr_std = []
    for neuron in states:
        firing_rates = []
        for rate in neuron:
            firing_rates.append(rate-np.min(neuron))
        fr_mean.append(np.mean(firing_rates))
        fr_std.append(np.std(firing_rates))
    return fr_mean, fr_std


def corr_plot(total_states):
    plts = len(total_states)
    rows = int(plts/3) + plts % 3
    nhide = (3 - (plts % 3)) % 3
    f, axs = plt.subplots(nrows=rows, ncols=3, figsize=(4*3, 4*rows))
    ax = axs.flatten()
    ind = 0
    corrs_mean = []
    corrs_std = []
    for states in total_states:
        corr = np.corrcoef(states)
        ax[ind].imshow(corr, aspect='auto')
        ind += 1
        corrs_mean.append(np.mean(corr))
        corrs_std.append(np.std(corr))
    h = -1
    for n in range(nhide):
        ax[h].axis('off')
        h -= 1
    return f, corrs_mean, corrs_std


def plot_results(axs, protocol, mean_states, std_states, corrs_mean,
                 corrs_std):
    indp = PRTCLS_IND_MAP[protocol]
    indps = np.random.uniform(indp-0.05, indp+0.05, [len(mean_states)])
    axs[0].errorbar(indp, np.mean(mean_states), np.std(mean_states),
                    marker='x', color=clrs[indp-1])
    axs[0].plot(indps, mean_states, alpha=0.5, linestyle='None', marker='x',
                color=clrs[indp-1])
    axs[1].errorbar(indp, np.mean(std_states), np.std(std_states), marker='x',
                    color=clrs[indp-1])
    axs[1].plot(indps, std_states, alpha=0.5, linestyle='None', marker='x',
                color=clrs[indp-1])
    axs[2].errorbar(indp, np.mean(corrs_mean), np.std(corrs_mean), marker='x',
                    color=clrs[indp-1])
    axs[2].plot(indps, corrs_mean, alpha=0.5, linestyle='None', marker='x',
                color=clrs[indp-1])
    axs[3].errorbar(indp, np.mean(corrs_std), np.std(corrs_std), marker='x',
                    color=clrs[indp-1])
    axs[3].plot(indps, corrs_std, alpha=0.5, linestyle='None', marker='x',
                color=clrs[indp-1])


if __name__ == "__main__":
    n_ch = [2]  # , 10, 20]
    alg = A2C
    protocols = ['01234', '4']
    for n in n_ch:
        folder = '/Users/martafradera/Desktop/models/'+str(n)+'_ch/'

        get_activity(folder, alg, protocols, n_ch=n, num_steps=200,
                     sv_model=True)
    # create_env('CVLearning-v0', 2)
