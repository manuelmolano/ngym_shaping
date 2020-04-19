#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting functions."""
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import gym
import seaborn as sns
import ntpath
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 7

clrs = sns.color_palette()

prtcls_index_map = {'01234': -1, '1234': 0, '0234': 1, '0134': 2, '0124': 3,
                    '34': 4}

ths_index_map = {'full': 0.5, '0.6': 0.6, '0.65': 0.65, '0.7': 0.7,
                 '0.75': 0.75, '0.8': 0.8, '0.85': 0.85, '0.9': 0.9}

all_indx = {}
all_indx.update(prtcls_index_map)
all_indx.update(ths_index_map)


def plot_env(env, num_steps_env=200, def_act=None, model=None, show_fig=True,
             name=None, legend=True, obs_traces=[], fig_kwargs={}):
    """
    env: already built neurogym task or name of it
    num_steps_env: number of steps to run the task
    def_act: if not None (and model=None), the task will be run with the
             specified action
    model: if not None, the task will be run with the actions predicted by
           model, which so far is assumed to be created and trained with the
           stable-baselines toolbox:
               (https://github.com/hill-a/stable-baselines)
    name: title to show on the rewards panel
    legend: whether to show the legend for actions panel or not.
    obs_traces: if != [] observations will be plot as traces, with the labels
                specified by obs_traces
    fig_kwargs: figure properties admited by matplotlib.pyplot.subplots() fun.
    """
    # We don't use monitor here because:
    # 1) env could be already prewrapped with monitor
    # 2) monitor will save data and so the function will need a folder
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations, obs_cum, rewards, actions, perf, actions_end_of_trial,\
        gt, states = run_env(env=env, num_steps_env=num_steps_env,
                             def_act=def_act, model=model)
    obs_cum = np.array(obs_cum)
    obs = np.array(observations)
    if show_fig:
        fig_(obs, actions, gt, rewards, legend=legend, performance=perf,
             states=states, name=name, obs_traces=obs_traces,
             fig_kwargs=fig_kwargs, env=env)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def run_env(env, num_steps_env=200, def_act=None, model=None):
    observations = []
    obs_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    obs = env.reset()
    obs_cum_temp = obs
    for stp in range(int(num_steps_env)):
        if model is not None:
            action, _states = model.predict(obs)
            if isinstance(action, float) or isinstance(action, int):
                action = [action]
            if len(_states) > 0:
                state_mat.append(_states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        obs_cum_temp += obs
        obs_cum.append(obs_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            obs_aux = obs[0]
            rew = rew[0]
            done = done[0]
            action = action[0]
        else:
            obs_aux = obs

        if done:
            env.reset()
        observations.append(obs_aux)
        if info['new_trial']:
            actions_end_of_trial.append(action)
            perf.append(info['performance'])
            obs_cum_temp = np.zeros_like(obs_cum_temp)
        else:
            actions_end_of_trial.append(-1)
            perf.append(-1)
        rewards.append(rew)
        actions.append(action)
        if 'gt' in info.keys():
            gt.append(info['gt'])
        else:
            gt.append(0)
    if model is not None and len(state_mat) > 0:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    return observations, obs_cum, rewards, actions, perf,\
        actions_end_of_trial, gt, states


def fig_(obs=None, actions=None, gt=None, rewards=None, states=None,
         performance=None, legend=True, obs_traces=None, name='', folder='',
         fig_kwargs={}, path=None, env=None, sv_data=False, start=None,
         end=None, show_delays=False, dash=None, show_perf=True,
         show_gt=True):
    """
    obs, actions: data to plot
    gt, rewards, states: if not None, data to plot
    mean_perf: mean performance to show in the rewards panel
    legend: whether to save the legend in actions panel
    folder: if != '', where to save the figure
    name: title to show on the rewards panel and name to save figure
    legend: whether to show the legend for actions panel or not.
    obs_traces: if != [] observations will be plot as traces, with the labels
                specified by obs_traces
    fig_kwargs: figure properties admited by matplotlib.pyplot.subplots() fun.
    """

    if path is not None:
        if start is None:
            start = 0
        if end is None:
            end = 100
        data = load_data(path)
        obs = data['obs'][start:end]
        actions = data['actions'][start:end]
        gt = data['gt'][start:end]
        rewards = data['rewards'][start:end]
        performance = data['performance'][start:end]

    obs = np.array(obs)
    actions = np.array(actions)
    if len(obs.shape) != 2:
        raise ValueError('obs has to be 2-dimensional.')
    steps = np.arange(obs.shape[0])

    n_row = 2  # observation and action
    n_row += rewards is not None
    n_row += states is not None

    gt_colors = 'gkmcry'
    if not fig_kwargs:
        fig_kwargs = dict(sharex=True, figsize=(5, n_row*1.5))

    f, axes = plt.subplots(n_row, 1, **fig_kwargs)

    # obs
    ax = axes[0]
    d = 0
    # duration of decision period
    d_start = []
    d_end = []

    if obs_traces:
        assert len(obs_traces) == obs.shape[1],\
            'Please provide label for each trace in the observations'
        for ind_tr, tr in enumerate(obs_traces):
            if ind_tr == dash:
                ax.plot(obs[:, ind_tr], '--', label=obs_traces[ind_tr])
            else:
                ax.plot(obs[:, ind_tr], label=obs_traces[ind_tr])
            # decision
            if ind_tr == 0:
                fixation = obs[:, ind_tr]
                for ind, action in enumerate(fixation):
                    # define first and last step of decision period
                    if action == 0 and d == 0:
                        d_start.append(ind)
                        d = 1
                    elif action == 1 and d == 1:
                        d_end.append(ind-1)
                        d = 0
                if len(d_start) > len(d_end):
                    d_end.append(end)
            elif ind_tr == 1:
                stim1 = obs[:, ind_tr]
            elif ind_tr == 2:
                stim2 = obs[:, ind_tr]

        stim = []
        for s1, s2 in zip(stim1, stim2):
            if s1 > 0 or s2 > 0:
                stim.append(1)
            else:
                stim.append(0)
        # delay
        dly = 0
        # duration of delay period
        # predly: fixation an delay periods
        predly_start = []
        predly_end = []
        for ind, value in enumerate(fixation):
            if value != stim[ind] and dly == 0:
                predly_start.append(ind)
                dly = 1
            elif value == stim[ind] and dly == 1:
                predly_end.append(ind-1)
                dly = 0
        if len(predly_start) > len(predly_end):
            predly_end.append(end)

        # dly period: previous than decision period
        dly_start = []
        dly_end = []
        for ind, step in enumerate(predly_end):
            if step+1 in d_start:
                dly_start.append(predly_start[ind])
                dly_end.append(predly_end[ind])

        ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])
    else:
        ax.imshow(obs.T, aspect='auto')
        if env and env.ob_dict:
            # Plot environment annotation
            yticks = []
            yticklabels = []
            for key, val in env.ob_dict.items():
                yticks.append((np.min(val)+np.max(val))/2)
                yticklabels.append(key)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_yticks([])

    if name:
        ax.set_title(name + ' env')
    ax.set_ylabel('Observations')

    # actions
    ax = axes[1]
    if len(actions.shape) > 1:
        # Changes not implemented yet
        ax.plot(steps, actions, marker='+', label='Actions')
    else:
        ax.plot(steps, actions, marker='+', label='Actions')

    # plot gt
    if gt is not None and show_gt is True:
        gt = np.array(gt)
        if len(gt.shape) > 1:
            for ind_gt in range(gt.shape[1]):
                ax.plot(steps, gt[:, ind_gt], '--'+gt_colors[ind_gt],
                        label='Ground truth '+str(ind_gt))
        else:
            ax.plot(steps, gt, '--'+gt_colors[0], label='Ground truth')
    ax.set_xlim([-0.5, len(steps)-0.5])
    ax.set_ylabel('Actions')
    if legend:
        ax.legend()
    if env and env.act_dict:
        # Plot environment annotation
        yticks = []
        yticklabels = []
        for key, val in env.act_dict.items():
            yticks.append((np.min(val) + np.max(val)) / 2)
            yticklabels.append(key)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    yticks = [0, 1, 2]
    yticklabels = ['Fixate', 'Left', 'Right']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # rewards
    if rewards is not None:
        ax = axes[2]
        ax.plot(steps, rewards, 'r', label='Rewards')
        if show_perf:
            # ind and perf value where perf != -1
            new_perf = []
            new_steps = []
            for ind, value in enumerate(performance):
                if value != -1:
                    new_perf.append(value)
                    new_steps.append(steps[ind])
            ax.plot(new_steps, new_perf, 'x', color='k', label='Performance')
            ax.set_ylabel('Reward/Performance')
        else:
            ax.set_ylabel('Reward')
        # performance = np.array(performance)
        # mean_perf = np.mean(performance[performance != -1])
        # ax.set_title('Mean performance: ' + str(np.round(mean_perf, 2)))
        if legend:
            ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])

        if env and env.rewards:
            # Plot environment annotation
            yticks = []
            yticklabels = []
            for key, val in env.rewards.items():
                yticks.append(val)
                yticklabels.append('{:s} {:0.2f}'.format(key, val))
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

    # states
    if states is not None:
        ax.set_xticks([])
        ax = axes[3]
        plt.imshow(states[:, int(states.shape[1]/2):].T,
                   aspect='auto')
        ax.set_title('Activity')
        ax.set_ylabel('Neurons')

    ax.set_xlabel('Steps')

    # plot decision area
    for ind, value in enumerate(d_start):
        if value == start:
            [ax.axvspan(d_start[ind], d_end[ind]+0.5, facecolor='grey',
                        alpha=0.3) for ax in axes]
        elif d_end[ind] == end:
            [ax.axvspan(d_start[ind]-0.5, end, facecolor='grey',
                        alpha=0.3) for ax in axes]
        else:
            [ax.axvspan(d_start[ind]-0.5, d_end[ind]+0.5, facecolor='grey',
                        alpha=0.3) for ax in axes]

    # plot delay area
    if show_delays:
        for ind, value in enumerate(dly_start):
            if value == start:
                [ax.axvspan(dly_start[ind], dly_end[ind]+0.5, facecolor='blue',
                            alpha=0.2) for ax in axes]
            elif d_end[ind] == end:
                [ax.axvspan(dly_start[ind]-0.5, end, facecolor='blue',
                            alpha=0.2) for ax in axes]
            else:
                [ax.axvspan(dly_start[ind]-0.5, dly_end[ind]+0.5,
                            facecolor='blue', alpha=0.2) for ax in axes]

    plt.tight_layout()

    if folder is not None and folder != '':
        if folder.endswith('.png') or folder.endswith('.svg'):
            f.savefig(folder)
        else:
            f.savefig(folder + name + 'env_struct.png')
        plt.close(f)

    return f


def data_extraction(folder, window=500, metrics={'reward': []}, conv=[1],
                    wind_final_perf=200, curr_perf=False):
    data = put_together_files(folder)
    data_flag = True
    if data:
        for ind_k, k in enumerate(metrics.keys()):
            ind_f = k.find('_final')
            if ind_f == -1:
                metric = data[k]
                if isinstance(window, float):
                    if window < 1.0:
                        window = int(metric.size * window)
                if conv[ind_k]:
                    mean = np.convolve(metric, np.ones((window,))/window,
                                       mode='valid')
                else:
                    mean = metric
                metrics[k].append(mean)
            elif k != 'curr_ph_final':
                metric = data[k[:ind_f]]
                metrics[k].append(np.mean(metric[-wind_final_perf:]))
        if curr_perf is True:
            metric = {'curr_perf': data['curr_perf']}
            metrics.update(metric)
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


def find_reaching_phase_time(trace, phase=4):
    trials = 1
    stop = False
    fphase = trace[0][-1]
    fstart = 0
    for curr_ph in trace[0]:
        if curr_ph != phase and stop is False:
            trials += 1
        else:
            stop = True
        if curr_ph != fphase:
            fstart += 1
    return trials, fphase, fstart


def put_together_files(folder):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    data = {}
    if len(files) > 0:
        files = order_by_sufix(files)
        file_data = np.load(files[0], allow_pickle=True)
        for key in file_data.keys():
            data[key] = file_data[key]

        for ind_f in range(1, len(files)):
            file_data = np.load(files[ind_f], allow_pickle=True)
            for key in file_data.keys():
                data[key] = np.concatenate((data[key], file_data[key]))
        np.savez(folder + '/bhvr_data_all.npz', **data)
    return data


def load_data(path):
    data = {}
    file_data = np.load(path, allow_pickle=True)
    for key in file_data.keys():
        data[key] = file_data[key]
    return data


def order_by_sufix(file_list):
    sfx = [int(x[x.rfind('_')+1:x.rfind('.')]) for x in file_list]
    sorted_list = [x for _, x in sorted(zip(sfx, file_list))]
    return sorted_list


def get_tag(tag, file):
    # process name
    f_name = ntpath.basename(file)
    assert f_name.find(tag) != -1, 'Tag not found'
    val = f_name[f_name.find(tag)+len(tag)+1:]
    val = val[:val.find('_')] if '_' in val else val
    if val == '-1.0':
        val = 'full'
    return val


def plot_results(folder, algorithm, w, wind_final_perf=100,
                 keys=['performance', 'curr_ph'], limit_ax=True, final_ph=4,
                 final_perf=0.7, ax_final=None, tag='th_stage', limit_tr=False,
                 f_final_prop={'color': (0, 0, 0), 'label': ''}, rerun=True):
    assert ('performance' in keys) and ('curr_ph' in keys),\
        'performance and curr_ph need to be included in the metrics (keys)'
    # load files
    if not os.path.exists(folder+'/data_'+algorithm+'_'+w+'.npz') or rerun:
        if tag == 'th_stage':
            files = glob.glob(folder + '*' + algorithm + '*' + w)
        else:
            files = glob.glob(folder + '*' + algorithm + '*stages*')
        files = sorted(files)
        val_index = []  # stores values for each instance
        metrics = {k: [] for k in keys}
        tmp = {k+'_final': [] for k in keys}
        metrics.update(tmp)
        tr_to_perf = []  # stores trials to reach final performance
        reached_ph = []  # stores whether the final phase is reached
        reached_perf = []  # stores whether the pre-defined perf is reached
        exp_durations = []
        stability_mat = []
        for ind_f, file in enumerate(files):
            val = get_tag(tag, file)
            # get metrics
            metrics, flag = data_extraction(folder=file, metrics=metrics,
                                            conv=[1, 0],
                                            wind_final_perf=wind_final_perf)
            if flag:
                # store durations
                exp_durations.append(len(metrics['curr_ph'][-1]))
                # store values
                val_index.append(val)
                # number of trials until final phase
                metrics, reached = tr_to_final_ph(metrics, wind_final_perf,
                                                  final_ph)
                reached_ph.append(reached)
                # number of trials until final perf
                tr_to_perf, reached =\
                    tr_to_reach_perf(metrics, reach_perf=final_perf,
                                     tr_to_perf=tr_to_perf,
                                     final_ph=final_ph)
                reached_perf.append(reached)
                # stability
                stability_mat.append(compute_stability(metrics=metrics,
                                                       tr_ab_th=tr_to_perf[-1]))
        val_index = np.array(val_index)
        metrics['val_index'] = val_index
        metrics['tr_to_perf'] = tr_to_perf
        metrics['reached_ph'] = reached_ph
        metrics['reached_perf'] = reached_perf
        metrics['exp_durations'] = exp_durations
        metrics['stability_mat'] = stability_mat
        np.savez(folder+'/data_'+algorithm+'_'+w+'_'+str(limit_tr)+'.npz',
                 **metrics)
    # plot results
    metrics = np.load(folder+'/data_'+algorithm+'_'+w+'_'+str(limit_tr)+'.npz',
                      allow_pickle=True)
    names = ['values_across_training_', 'mean_values_across_training_']
    val_index = metrics['val_index']
    for ind in range(2):
        f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                             figsize=(6, 6))
        # plot means
        for ind_met, met in enumerate(keys):
            metric = metrics[met]
            if ind == 0:
                plot_rew_across_training(metric=metric, index=val_index,
                                         ax=ax[ind_met], clrs=clrs)
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], clrs=clrs, limit_ax=limit_ax)
            elif ind == 1:
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], clrs=clrs, limit_ax=limit_ax)
        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[0].set_ylabel('Average performance')
        ax[1].set_ylabel('Average phase')
        ax[1].set_xlabel('Trials')
        ax[1].legend()
        f.savefig(folder+'/'+names[ind]+algorithm+'_'+w+'_'+str(limit_tr)+'.png',
                  dpi=200)
        plt.close(f)
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    ax.plot(metrics['exp_durations'], metrics['stability_mat'], '+')
    corr_ = np.corrcoef(metrics['exp_durations'], metrics['stability_mat'])
    ax.set_title('Correlation: '+str(np.round(corr_[0, 1], 2)))
    f.savefig(folder+'/corr_stablty_dur'+algorithm+'_'+w+'_' +
              str(limit_tr)+'.png', dpi=200)
    plt.close(f)
    # define xticks
    if tag == 'stages':
        labels = list(prtcls_index_map.keys())
        ticks = list(prtcls_index_map.values())
    elif tag == 'th_stage':
        labels = list(ths_index_map.keys())
        ticks = list(ths_index_map.values())
    # plot final results
    if ax_final is not None:
        # plot final performance
        plt_perf_indicators(values=metrics['performance_final'],
                            reached=metrics['reached_ph'],
                            f_props=f_final_prop, index_val=val_index,
                            ax=ax_final[0, 0])
        ax_final[0, 0].set_xlabel(tag)
        ax_final[0, 0].set_ylabel('Average performance')
        ax_final[0, 0].set_xticks(ticks)
        ax_final[0, 0].set_xticklabels(labels)
        # trials to reach phase 4
        plt_perf_indicators(values=metrics['curr_ph_final'],
                            f_props=f_final_prop,
                            index_val=val_index, ax=ax_final[0, 1],
                            reached=metrics['reached_ph'], discard=['full'])
        ax_final[0, 1].set_xlabel(tag)
        ax_final[0, 1].set_ylabel('Number of trials to reach phase 4')
        ax_final[0, 1].set_xticks(ticks)
        ax_final[0, 1].set_xticklabels(labels)
        # trials to reach final perf
        plt_perf_indicators(values=metrics['tr_to_perf'],
                            reached=metrics['reached_perf'],
                            index_val=val_index, ax=ax_final[0, 2],
                            f_props=f_final_prop)
        ax_final[0, 2].set_xlabel(tag)
        ax_final[0, 2].set_ylabel('Number of trials to reach' +
                                  ' final performance')
        ax_final[0, 2].set_xticks(ticks)
        ax_final[0, 2].set_xticklabels(labels)
        # make -1s equal to total number of trials
        plt_perf_indicators(values=metrics['reached_ph'],
                            index_val=val_index, ax=ax_final[1, 0],
                            f_props=f_final_prop, discard=['full'],
                            errorbars=False, plot_individual_values=False)
        ax_final[1, 0].set_xlabel(tag)
        ax_final[1, 0].set_ylabel('Proportion of instances reaching phase 4')
        ax_final[1, 0].set_xticks(ticks)
        ax_final[1, 0].set_xticklabels(labels)
        # prop of trials that reach final perf
        plt_perf_indicators(values=metrics['reached_perf'],
                            index_val=val_index, ax=ax_final[1, 1],
                            f_props=f_final_prop, errorbars=False,
                            plot_individual_values=False)
        ax_final[1, 1].set_xlabel(tag)
        ax_final[1, 1].set_ylabel('Proportion of instances reaching' +
                                  ' final perf')
        ax_final[1, 1].set_xticks(ticks)
        ax_final[1, 1].set_xticklabels(labels)
        # plot stability
        plt_perf_indicators(values=metrics['stability_mat'],
                            index_val=val_index,  ax=ax_final[1, 2],
                            f_props=f_final_prop,
                            reached=metrics['reached_perf'])
        ax_final[1, 2].set_xlabel(tag)
        ax_final[1, 2].set_ylabel('Stability')
        ax_final[1, 2].set_xticks(ticks)
        ax_final[1, 2].set_xticklabels(labels)


def tr_to_final_ph(metrics, wind_final_perf, final_ph):
    curr_ph = metrics['curr_ph'][-1]
    time = np.where(curr_ph == final_ph)[0]  # find those trials in phase 4
    reached = False
    if len(time) != 0:
        first_tr = np.min(time)  # min trial is first trial in phase 4
        if first_tr > len(curr_ph) - wind_final_perf:
            # if phase 4 is not reached, last trial is obtained
            metrics['curr_ph_final'].append(len(curr_ph))
        else:
            metrics['curr_ph_final'].append(first_tr)
            reached = True
    else:
        metrics['curr_ph_final'].append(len(curr_ph))
    return metrics, reached


def tr_to_reach_perf(metrics, reach_perf, tr_to_perf, final_ph):
    perf = np.array(metrics['performance'][-1])
    curr_ph = metrics['curr_ph'][-1]
    time_final_ph = np.where(curr_ph == final_ph)[0]
    reached = False
    if len(time_final_ph) == 0:
        tr_to_perf.append(len(perf))
    else:
        tr_to_ph = np.min(time_final_ph)
        perf_in_final_ph = perf[tr_to_ph:]
        time_above_th = np.where(perf_in_final_ph > reach_perf)[0]
        if len(time_above_th) == 0:
            tr_to_perf.append(len(perf))
        else:
            reached = True
            tr_to_perf.append(np.min(time_above_th) +
                              np.min(time_final_ph))
    return tr_to_perf, reached


def compute_stability(metrics, tr_ab_th):
    perf = np.array(metrics['performance'][-1])[tr_ab_th:]
    forgetting_times = perf < 0.5
    stability = 1 - np.sum(forgetting_times)/perf.shape[0]
    return stability


def plot_rew_across_training(metric, index, ax, clrs):
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx]
        for trace in traces_temp:
            ax.plot(trace, color=clrs[ind_val], alpha=0.5, lw=0.5)


def plt_means(metric, index, ax, clrs, limit_mean=True, limit_ax=True):
    if limit_mean:
        min_dur = np.min([len(x) for x in metric])
        metric = [x[:min_dur] for x in metric]
    else:
        max_dur = np.max([len(x) for x in metric])
        metric = [np.concatenate((np.array(x),
                                  np.nan*np.ones((int(max_dur-len(x)),))))
                  for x in metric]

    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx, :]
        ax.plot(np.nanmean(traces_temp, axis=0), color=clrs[ind_val],
                lw=1, label=val+' ('+str(np.sum(indx))+')')
    if limit_ax:
        assert limit_mean, 'limiting ax only works when mean is also limited'
        ax.set_xlim([0, min_dur])


def get_noise(unq_vals):
    max_ = np.max([all_indx[x] for x in unq_vals])
    min_ = np.min([all_indx[x] for x in unq_vals])
    noise = (max_ - min_)/40
    return noise


def plt_perf_indicators(values, index_val, ax, f_props, reached=None,
                        discard=[], plot_individual_values=True,
                        errorbars=True):
    values = np.array(values)  # tr until final phase
    index_val = np.array(index_val)
    unq_vals = np.unique(index_val)
    if plot_individual_values:
        std_noise = get_noise(unq_vals)
    for ind_val, val in enumerate(unq_vals):
        # only for those thresholds different than full task
        if val not in discard:
            # only those traces with same value that have reached last phase
            if reached is not None:
                indx = np.logical_and(index_val == val, reached)
            else:
                indx = index_val == val
            values_temp = values[indx]
            n_vals = len(values_temp)
            if n_vals != 0:
                # plot number of trials
                f_props['markersize'] = 6
                if errorbars:
                    ax.errorbar([all_indx[val]], np.nanmean(values_temp),
                                (np.std(values_temp)/np.sqrt(n_vals)),
                                **f_props)
                else:
                    ax.plot(all_indx[val], np.nanmean(values_temp), **f_props)
            if plot_individual_values:
                xs = np.random.normal(0, std_noise, ((np.sum(indx),))) +\
                    all_indx[val]
                ax.plot(xs, values_temp, marker=f_props['marker'],
                        color=f_props['color'], alpha=0.5, linestyle='None')


def process_results_diff_thresholds(folder):
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    windows = ['0', '1', '2', '3', '4']  # , '500', '1000']
    markers = ['+', 'x', '1', 'o', '>']
    for alg in algs:
        print(alg)
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        ind = 0
        for ind_w, w in enumerate(windows):
            print('xxxxxxxxxxxxxxxxxxxxxxxx')
            print('Window')
            print(w)
            marker = markers[ind]
            ind += 1
            plot_results(folder, alg, w, limit_ax=False,
                         ax_final=ax, tag='th_stage',
                         f_final_prop={'color': clrs[ind_w],
                                       'label': str(w),
                                       'marker': marker})

        handles, labels = ax[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0, 0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+'.png', dpi=200)
        plt.close(f)


def process_results_diff_protocols(folder):
    algs = ['A2C']   # , 'ACER', 'PPO2', 'ACKTR']
    w = '0'
    marker = '+'
    for alg in algs:
        print(alg)
        print('xxxxxxxxxxxxxxxxxxxxxx')
        f, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        plot_results(folder, alg, w, limit_ax=False,
                     ax_final=ax, tag='stages', limit_tr=None,
                     f_final_prop={'color': clrs[0],
                                   'label': w,
                                   'marker': marker})

        handles, labels = ax[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0, 0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+'.png', dpi=200)
        plt.close(f)


if __name__ == '__main__':
    plt.close('all')
    # folder = '/Users/martafradera/Desktop/OneDrive -' +\
    #          ' Universitat de Barcelona/TFG/task/data/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/RL_algs/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/one_agent_control/'
    folder = '/home/manuel/CV-Learning/results/results_2303/diff_protocols/'
    # folder = '/gpfs/projects/hcli64/shaping/diff_protocols/'
    # process_results_diff_protocols(folder)
    # folder = '/gpfs/projects/hcli64/shaping/one_agent_control/'
    process_results_diff_protocols(folder)
