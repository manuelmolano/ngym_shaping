#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting functions."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import gym
import seaborn as sns
import ntpath
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12

CLRS = sns.color_palette()

STAGES = [0, 1, 2, 3, 4]

PRTCLS_IND_MAP = {'01234': -1, '1234': 0, '0234': 1, '0134': 2, '0124': 3,
                  '34': 4, 'full': 5}

THS_IND_MAP = {'full': 0.5, '0.6': 0.6, '0.65': 0.65, '0.7': 0.7,
               '0.75': 0.75, '0.8': 0.8, '0.85': 0.85, '0.9': 0.9}

ALL_INDX = {}
ALL_INDX.update(PRTCLS_IND_MAP)
ALL_INDX.update(THS_IND_MAP)


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


def load_data(path):
    data = {}
    file_data = np.load(path, allow_pickle=True)
    for key in file_data.keys():
        data[key] = file_data[key]
    return data


def data_extraction(folder, w_conv_perf=500, metrics={'reward': []},
                    conv=[1]):
    data = put_together_files(folder)
    data_flag = True
    if data:
        for ind_k, k in enumerate(metrics.keys()):
            if k in data.keys():
                metric = data[k]
                if conv[ind_k]:
                    mean = np.convolve(metric,
                                       np.ones((w_conv_perf,))/w_conv_perf,
                                       mode='valid')
                else:
                    mean = metric
            else:
                mean = []
            metrics[k].append(mean)
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


def perf_hist(metric, ax, index, trials_day=300):
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    bins = np.linspace(0, 1, 20)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx].flatten()
        hist_, plt_bins = np.histogram(traces_temp, bins=bins)
        hist_ = hist_/np.sum(hist_)
        plt_bins = plt_bins[:-1] + (plt_bins[1]-plt_bins[0])/2
        ax.plot(plt_bins, hist_, label=val, color=CLRS[ind_val])
    ax.legend()
    ax.set_xlabel('Performance')
    ax.set_ylabel('Days')


def trials_per_stage(metric, ax, index):
    bins = np.linspace(STAGES[0]-0.5, STAGES[-2]+.5, len(STAGES))
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx]
        counts_mat = []
        n_traces = len(traces_temp)
        if val == '0234':
            plt.figure()
            for ind in range(n_traces):
                plt.plot(traces_temp[ind]+4*ind)
            # print('asdas')
        for ind_tr in range(n_traces):
            counts = np.histogram(traces_temp[ind_tr], bins=bins)[0]
            # ax.plot(STAGES[:-1]+np.random.normal(0, 0.01, 4), counts, '+',
            #         color=CLRS[ind_val], alpha=0.5)
            counts_mat.append(counts)
        counts_mat = np.array(counts_mat)
        mean_counts = np.mean(counts_mat, axis=0)
        std_counts = np.std(counts_mat, axis=0)/np.sqrt(n_traces)
        # TODO: don't plot stages that are never visited
        # (e.g. in protocol 0234, don't plot stage 1)
        ax.errorbar(np.array(STAGES[:-1]), mean_counts, std_counts, marker='+',
                    color=CLRS[ind_val], label=val)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Stage')
    ax.set_ylabel('Trials')


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
    if val.find('-1') != -1:
        val = 'full'
    return val


def plot_results(folder, algorithm, w, w_conv_perf=500,
                 keys=['performance', 'curr_ph', 'num_stps', 'curr_perf'],
                 limit_ax=True, final_ph=4, perf_th=0.7, ax_final=None,
                 tag='th_stage', limit_tr=False, rerun=False,
                 f_final_prop={'color': (0, 0, 0), 'label': ''},
                 plt_ind_vals=True, plt_all_traces=False):
    assert ('performance' in keys) and ('curr_ph' in keys),\
        'performance and curr_ph need to be included in the metrics (keys)'
    # PROCESS RAW DATA
    if not os.path.exists(folder+'/data_'+algorithm+'_'+w+'.npz') or rerun:
        if tag == 'th_stage':
            files = glob.glob(folder + '*' + algorithm + '*' + w)
        else:
            files = glob.glob(folder + '*' + algorithm + '*stages*')
        files = sorted(files)
        val_index = []  # stores values for each instance
        metrics = {k: [] for k in keys}
        keys = np.array(keys)
        for ind_f, file in enumerate(files):
            val = get_tag(tag, file)
            # get metrics
            metrics, flag = data_extraction(folder=file, metrics=metrics,
                                            w_conv_perf=w_conv_perf,
                                            conv=[1, 0, 1, 0])
            # store values
            val_index.append(val)
        metrics['val_index'] = np.array(val_index)
        np.savez(folder+'/data_'+algorithm+'_'+w+'.npz', **metrics)

    # LOAD AND (POST)PROCESS DATA
    print('Loading data')
    tmp = np.load(folder+'/data_'+algorithm+'_'+w+'.npz', allow_pickle=True)
    # the loaded file does not allow to modifying it
    metrics = {}
    for k in tmp.keys():
        metrics[k] = list(tmp[k])
    if limit_tr:
        min_dur = np.min([len(x) for x in metrics['curr_ph']])
    else:
        min_dur = np.max([len(x) for x in metrics['curr_ph']])
    tr_to_perf = []  # stores trials to reach final performance
    reached_ph = []  # stores whether the final phase is reached
    reached_perf = []  # stores whether the pre-defined perf is reached
    exp_durations = []
    stability_mat = []
    final_perf = []
    tr_to_ph = []
    stps_to_perf = []
    stps_to_ph = []
    for ind_f in range(len(metrics['curr_ph'])):
        # store durations
        exp_durations.append(len(metrics['curr_ph'][ind_f]))
        for k in metrics.keys():
            metrics[k][ind_f] = metrics[k][ind_f][:min_dur]
            if len(metrics[k][ind_f]) == 0:
                metrics[k][ind_f] = np.nan*np.ones((min_dur,))
        # phase analysis
        curr_ph = metrics['curr_ph'][ind_f]
        # number of trials until final phase
        tr_to_ph, reached = tr_to_final_ph(curr_ph, tr_to_ph, w_conv_perf,
                                           final_ph)
        reached_ph.append(reached)
        # performance analysis
        perf = np.array(metrics['performance'][ind_f])
        # get final performance
        final_perf.append(perf[-1])
        # get trials to reach specified performance
        tt_ph = tr_to_ph[-1]
        tr_to_perf, reached = tr_to_reach_perf(perf=perf.copy(), tr_to_ph=tt_ph,
                                               reach_perf=perf_th,
                                               tr_to_perf=tr_to_perf,
                                               final_ph=final_ph)
        reached_perf.append(reached)
        # performance stability
        tt_prf = tr_to_perf[-1]
        stability_mat.append(compute_stability(perf=perf.copy(),
                                               tr_ab_th=tt_prf))
        # # number of steps
        if len(metrics['num_stps'][ind_f]) != 0:
            num_steps = np.cumsum(metrics['num_stps'][ind_f])
            stps_to_perf.append(num_steps[tt_prf-1]/1000)
            stps_to_ph.append(num_steps[tt_ph-1]/1000)
        else:
            stps_to_perf.append(np.nan)
            stps_to_ph.append(np.nan)
    print('Plotting results')
    # define xticks
    ax_props = {'tag': tag}
    if tag == 'stages':
        ALL_INDX['full'] = 5
        ax_props['labels'] = list(PRTCLS_IND_MAP.keys())
        ax_props['ticks'] = list(PRTCLS_IND_MAP.values())
    elif tag == 'th_stage':
        ax_props['labels'] = list(THS_IND_MAP.keys())
        ax_props['ticks'] = list(THS_IND_MAP.values())

    # plot results
    names = ['values_across_training_', 'mean_values_across_training_']
    ylabels = ['Performance', 'Phase', 'Number of steps', 'Session performance']
    val_index = metrics['val_index']
    for ind in range(2):
        f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                             figsize=(12, 12))
        # plot means
        for ind_met, met in enumerate(keys):
            metric = metrics[met]
            if ind == 0 and plt_all_traces:
                plot_rew_across_training(metric=metric, index=val_index,
                                         ax=ax[ind_met])
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], limit_ax=limit_ax)
            elif ind == 1:
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], limit_ax=limit_ax)
            ax[ind_met].set_ylabel(ylabels[ind_met])
        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[len(keys)-1].set_xlabel('Trials')
        ax[len(keys)-1].legend()
        f.savefig(folder+'/'+names[ind]+algorithm+'_'+w+'_'+str(limit_tr)+'.png',
                  dpi=200)
        plt.close(f)

    # days under perf
    if 'curr_perf' in keys:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
        metric = metrics['curr_perf']
        perf_hist(metric, ax=ax, index=val_index, trials_day=300)
        ax.set_title('Performance histogram ('+algorithm+')')
        f.savefig(folder+'/perf_hist_'+algorithm+'_'+w+'.png', dpi=200)
        plt.close(f)

    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    ax.plot(exp_durations, stability_mat, '+')
    corr_ = np.corrcoef(exp_durations, stability_mat)
    ax.set_title('Correlation: '+str(np.round(corr_[0, 1], 2)))
    f.savefig(folder+'/corr_stablty_dur'+algorithm+'_'+w+'_' +
              str(limit_tr)+'.png', dpi=200)
    plt.close(f)

    # trials per stage
    if 'curr_ph' in keys:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        metric = metrics['curr_ph']
        trials_per_stage(metric, ax=ax, index=val_index)
        ax.set_title('Average number of trials per stage ('+algorithm+')')
        f.savefig(folder+'/trials_stage_'+algorithm+'_'+w+'.png', dpi=200)
        plt.close(f)

    # plot final results
    if ax_final is not None:
        # plot final performance
        ax_props['ylabel'] = 'Average performance'
        plt_perf_indicators(values=final_perf,
                            reached=reached_ph,
                            f_props=f_final_prop, index_val=val_index,
                            ax=ax_final[0], ax_props=ax_props,
                            plot_individual_values=plt_ind_vals)
        # prop of trials that reach final perf
        ax_props['ylabel'] = 'Proportion of instances reaching final perf'
        plt_perf_indicators(values=reached_perf, index_val=val_index,
                            ax=ax_final[1], f_props=f_final_prop,
                            ax_props=ax_props, errorbars=False,
                            plot_individual_values=False)
        # trials to reach final perf
        ax_props['ylabel'] = 'Number of trials to reach final performance'
        plt_perf_indicators(values=tr_to_perf,
                            reached=reached_perf,
                            index_val=val_index, ax=ax_final[2],
                            f_props=f_final_prop, ax_props=ax_props,
                            plot_individual_values=plt_ind_vals)
        # plot stability
        ax_props['ylabel'] = 'Stability'
        plt_perf_indicators(values=stability_mat, index_val=val_index,
                            ax=ax_final[3], f_props=f_final_prop,
                            ax_props=ax_props, reached=reached_perf,
                            plot_individual_values=plt_ind_vals)

        # make -1s equal to total number of trials
        ax_props['ylabel'] = 'Proportion of instances reaching phase 4'
        plt_perf_indicators(values=reached_ph, index_val=val_index,
                            ax=ax_final[4], f_props=f_final_prop,
                            ax_props=ax_props, discard=['full'],
                            errorbars=False, plot_individual_values=False)
        # trials to reach phase 4
        ax_props['ylabel'] = 'Number of trials to reach phase 4'
        plt_perf_indicators(values=tr_to_ph,
                            f_props=f_final_prop, ax_props=ax_props,
                            index_val=val_index, ax=ax_final[5],
                            reached=reached_ph, discard=['full'],
                            plot_individual_values=plt_ind_vals)
        # steps to reach phase 4
        ax_props['ylabel'] = 'Number of steps to reach phase 4'
        plt_perf_indicators(values=stps_to_ph,
                            f_props=f_final_prop, ax_props=ax_props,
                            index_val=val_index, ax=ax_final[6],
                            reached=reached_ph, discard=['full'],
                            plot_individual_values=plt_ind_vals)
        # steps to reach final perf
        ax_props['ylabel'] = 'Number of steps to reach final performance'
        plt_perf_indicators(values=stps_to_perf,
                            reached=reached_perf,
                            index_val=val_index, ax=ax_final[7],
                            f_props=f_final_prop, ax_props=ax_props,
                            plot_individual_values=plt_ind_vals)


def tr_to_final_ph(curr_ph, tr_to_ph, wind_final_perf, final_ph):
    time = np.where(curr_ph == final_ph)[0]  # find those trials in phase 4
    reached = False
    if len(time) != 0:
        first_tr = np.min(time)  # min trial is first trial in phase 4
        if first_tr > len(curr_ph) - wind_final_perf:
            # if phase 4 is not reached, last trial is obtained
            tr_to_ph.append(len(curr_ph))
        else:
            tr_to_ph.append(first_tr)
            reached = True
    else:
        tr_to_ph.append(len(curr_ph))
    return tr_to_ph, reached


def tr_to_reach_perf(perf, tr_to_ph, reach_perf, tr_to_perf, final_ph):
    reached = False
    perf_in_final_ph = perf[tr_to_ph:]
    time_above_th = np.where(perf_in_final_ph > reach_perf)[0]
    if len(time_above_th) == 0:
        tr_to_perf.append(len(perf))
    else:
        reached = True
        tr_to_perf.append(np.min(time_above_th) +
                          np.min(tr_to_ph))
    return tr_to_perf, reached


def compute_stability(perf, tr_ab_th):
    perf = np.array(perf)[tr_ab_th:]
    if perf.shape[0] != 0:
        forgetting_times = perf < 0.5
        stability = 1 - np.sum(forgetting_times)/perf.shape[0]
    else:
        stability = np.nan
    return stability


def plot_rew_across_training(metric, index, ax):
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx]
        for trace in traces_temp:
            ax.plot(trace, color=CLRS[ind_val], alpha=0.5, lw=0.5)


def plt_means(metric, index, ax, limit_mean=True, limit_ax=True):
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
        if not (np.isnan(traces_temp)).all():
            ax.plot(np.nanmean(traces_temp, axis=0), color=CLRS[ind_val],
                    lw=1, label=val+' ('+str(np.sum(indx))+')')
    if limit_ax:
        assert limit_mean, 'limiting ax only works when mean is also limited'
        ax.set_xlim([0, min_dur])


def get_noise(unq_vals):
    max_ = np.max([ALL_INDX[x] for x in unq_vals])
    min_ = np.min([ALL_INDX[x] for x in unq_vals])
    noise = (max_ - min_)/40
    return noise


def plt_perf_indicators(values, index_val, ax, f_props, ax_props, reached=None,
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
                f_props['markersize'] = 10
                if errorbars:
                    ax.errorbar([ALL_INDX[val]], np.nanmean(values_temp),
                                (np.nanstd(values_temp)/np.sqrt(n_vals)),
                                **f_props)
                else:
                    ax.plot(ALL_INDX[val], np.nanmean(values_temp), **f_props)
            if plot_individual_values:
                xs = np.random.normal(0, std_noise, ((np.sum(indx),))) +\
                    ALL_INDX[val]
                ax.plot(xs, values_temp, alpha=0.5, linestyle='None', **f_props)
    ax.set_xlabel(ax_props['tag'])
    ax.set_ylabel(ax_props['ylabel'])
    ax.set_xticks(ax_props['ticks'])
    ax.set_xticklabels(ax_props['labels'])


def process_results_diff_thresholds(folder, limit_tr=True):
    algs = ['PPO2', 'ACKTR', 'A2C', 'ACER']
    windows = ['0', '1', '2', '3', '4']  # , '500', '1000']
    markers = ['+', 'x', '1', 'o', '>']
    for alg in algs:
        print(alg)
        f, ax = plt.subplots(nrows=2, ncols=4, figsize=(27, 16))
        ax = ax.flatten()
        ind = 0
        for ind_w, w in enumerate(windows):
            print('xxxxxxxxxxxxxxxxxxxxxxxx')
            print('Window')
            print(w)
            marker = markers[ind]
            ind += 1
            plot_results(folder, alg, w, limit_ax=False, plt_ind_vals=False,
                         ax_final=ax, tag='th_stage', limit_tr=limit_tr,
                         f_final_prop={'color': CLRS[ind_w],
                                       'label': str(w),
                                       'marker': marker})

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+str(limit_tr)+'.png', dpi=200)
        plt.close(f)


def process_results_diff_protocols(folder, limit_tr=True):
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    w = '0'
    marker = '+'
    for alg in algs:
        print(alg)
        print('xxxxxxxxxxxxxxxxxxxxxx')
        f, ax = plt.subplots(nrows=2, ncols=4, figsize=(30, 16))
        ax = ax.flatten()
        plot_results(folder, alg, w, limit_ax=False,
                     ax_final=ax, tag='stages', limit_tr=limit_tr,
                     f_final_prop={'color': CLRS[0],
                                   'label': w,
                                   'marker': marker})

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+str(limit_tr)+'.png', dpi=200)
        plt.close(f)


if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) == 1:
        # folder = '/Users/martafradera/Desktop/OneDrive -' +\
        #     ' Universitat de Barcelona/TFG/task/data/'
        folder = '/home/manuel/CV-Learning/results/results_2303/'
    elif len(sys.argv) == 2:
        folder = sys.argv[1]
    print(sys.argv)
    # folder = '/home/manuel/CV-Learning/results/results_2303/RL_algs/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/one_agent_control/'

    # folder = '/gpfs/projects/hcli64/shaping/diff_protocols/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/one_agent_control/'
    # folder = '/gpfs/projects/hcli64/shaping/one_agent_control/'

    process_results_diff_protocols(folder+'/diff_protocols/', limit_tr=True)
    # process_results_diff_thresholds(folder+'/one_agent_control/',
    #                                 limit_tr=True)
