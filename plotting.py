#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting functions."""

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


def plot_rew_across_training(folder, window=100, ax=None, ytitle='', xlbl='',
                             metrics={'reward': []}, fkwargs={'c': 'tab:blue'},
                             legend=False, conv=[1], wind_final_perf=200):
    data = put_together_files(folder)
    data_flag = True
    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(nrows=len(metrics.keys()), ncols=1,
                                 figsize=(6, 6))
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
                ax[ind_k].plot(mean, **fkwargs)  # add color, label etc.
                ax[ind_k].set_xlabel(xlbl)
                # figure props
                if not ytitle:
                    ax[ind_k].set_ylabel('mean ' + k)
                else:
                    ax[ind_k].set_ylabel(ytitle)
                if legend:
                    ax[ind_k].legend()
                if ind_k == len(metrics.keys())-1:
                    ax[ind_k].set_xlabel('trials')
            elif k != 'curr_ph_final':
                metric = data[k[:ind_f]]
                metrics[k].append(np.mean(metric[-wind_final_perf:]))
        if sv_fig:
            f.savefig(folder + '/mean_reward_across_training.png')
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
    th = f_name[f_name.find(tag)+len(tag)+1:]
    th = th[:th.find('_')] if '_' in th else th
    new_th = -1
    if tag == 'stages':
        for stage in range(4):
            if th.find(str(stage)) == -1:
                new_th = stage
    else:
        new_th = th
    th = float(new_th)
    return th


def plot_results(folder, algorithm, w, marker, wind_final_perf=100,
                 keys=['performance', 'curr_ph'], limit_ax=True, final_ph=4,
                 final_perf=0.75, ax_final=None, tag='th_stage',
                 f_final_prop={'color': (0, 0, 0), 'label': ''}):
    assert ('performance' in keys) and ('curr_ph' in keys),\
        'performance and curr_ph need to be included in the metrics (keys)'
    # load files
    if tag == 'th_stage':
        files = glob.glob(folder + '*_' + algorithm + '_*_' + w)
    else:
        files = glob.glob(folder + '*_' + algorithm + '_*_stages_*')
    files = sorted(files)
    f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                         figsize=(6, 6))
    ths_mat = []  # stores unique thresholds
    ths_count = []
    th_index = []  # stores thresholds for each instance
    metrics = {k: [] for k in keys}
    tmp = {k+'_final': [] for k in keys}
    metrics.update(tmp)
    tr_to_final_perf = []
    reached_ph = []
    for ind_f, file in enumerate(files):
        th = get_tag(tag, file)  # TODO: can this process the case in which all stages are used?
        # check if th was already visited to assign color
        if th in ths_mat:
            ci = np.where(np.array(ths_mat) == th)[0][0]
            ths_count[ci] += 1
        else:
            ths_mat.append(th)
            ths_count.append(1)
            ci = len(ths_mat)-1
        # get and plot metrics
        metrics, flag = plot_rew_across_training(folder=file, ax=ax,
                                                 metrics=metrics,
                                                 conv=[1, 0],
                                                 wind_final_perf=wind_final_perf,
                                                 fkwargs={'c': clrs[ci],
                                                          'lw': 0.5,
                                                          'alpha': 0.5})
        if flag:
            th_index.append(th)
            # number of trials until final phase
            metrics, reached = tr_to_final_ph(metrics, wind_final_perf,
                                              final_ph)
            reached_ph.append(reached)
            # number of trials until final perf
            metrics = tr_to_reach_perf(metrics, reach_perf=final_perf,
                                       tr_to_final_perf=tr_to_final_perf,
                                       final_ph=final_ph)
    th_index = np.array(th_index)
    if metrics[keys[0]]:
        # plot means
        for ind_met, met in enumerate(metrics.keys()):
            ind_f = met.find('_final')
            if ind_f == -1:
                plt_means(metric=metrics[met], index=th_index, ax=ax[ind_met],
                          clrs=clrs, limit_ax=limit_ax)
        np.array
        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[0].set_ylabel('Average performance')
        ax[1].set_ylabel('Average phase')
        ax[1].set_xlabel('Trials')
        ax[1].legend()
        f.savefig(folder+'/values_across_training_'+algorithm+'_'+w+'.png',
                  dpi=200)
        plt.close(f)
        # plot only means
        f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                             figsize=(6, 6))
        for ind_met, met in enumerate(metrics.keys()):
            ind_f = met.find('_final')
            if ind_f == -1:
                plt_means(metric=metrics[met], index=th_index, ax=ax[ind_met],
                          clrs=clrs, limit_ax=limit_ax)

        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[0].set_ylabel('Average performance')
        ax[1].set_ylabel('Average phase')
        ax[1].set_xlabel('Trials')
        ax[1].legend()
        f.savefig(folder + '/mean_values_across_training_' +
                  algorithm+'_'+w+'.png', dpi=200)
        plt.close(f)
        # plot final results
        if ax_final is not None:
            # plot final performance
            plt_final_perf(final_perf=metrics['performance_final'],
                           marker=marker, reached_ph=reached_ph,
                           f_props=f_final_prop, index_th=th_index,
                           ax=ax_final[0, 0])
            ax_final[0, 0].set_xlabel('threshold')
            ax_final[0, 0].set_ylabel('Average performance')
            # trials to reach phase 4
            plt_final_tr_to_ph(tr_to_final_ph=metrics['curr_ph_final'],
                               marker=marker, f_props=f_final_prop,
                               index_th=th_index, ax=ax_final[0, 1], tag=tag)
            ax_final[0, 1].set_xlabel('threshold')
            ax_final[0, 1].set_ylabel('Number of trials to reach phase 4')
            # trials to reach final perf
            plt_tr_to_final_perf(tr_to_reach_perf=tr_to_final_perf,
                                 index_th=th_index, ax=ax_final[1, 0],
                                 f_props=f_final_prop, marker=marker)
            ax_final[1, 0].set_xlabel('threshold')
            ax_final[1, 0].set_ylabel('Number of trials to reach' +
                                      ' final performance')
            # make -1s equal to total number of trials
            prop_of_exp_reaching_ph(reached_ph=reached_ph, tag=tag,
                                    index_th=th_index, marker=marker,
                                    ax=ax_final[1, 1], f_props=f_final_prop)
            ax_final[1, 1].set_xlabel('threshold')
            ax_final[1, 1].set_ylabel('Proportion of instances reaching phase 4')
    else:
        plt.close(f)


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


def tr_to_reach_perf(metrics, reach_perf, tr_to_final_perf, final_ph):  # TODO: take only those that reach final phase
    perf = metrics['performance'][-1]
    # find those trials which performance is over reach perf
    time = np.where(perf >= reach_perf)[0]
    curr_ph = metrics['curr_ph'][-1]
    trials = []
    for value in time:
        if curr_ph[value] == final_ph:
            # only keep those trials which are in the last phase
            trials.append(value)
    if len(trials) != 0:
        # if last phase is not reached, last trial is obtained
        first_tr = np.min(trials)
        tr_to_final_perf.append(first_tr)
    else:
        tr_to_final_perf.append(len(perf))
    return metrics


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
    unq_ths = np.unique(index)
    for ind_th, th in enumerate(unq_ths):
        indx = index == th
        traces_temp = metric[indx, :]
        th_str = 'full' if ((th+1) < 0.01) else str(th)
        ax.plot(np.nanmean(traces_temp, axis=0), color=clrs[ind_th],
                lw=1, label=th_str+'('+str(np.sum(indx))+')')
    if limit_ax:
        assert limit_mean, 'limiting ax only works when mean is also limited'
        ax.set_xlim([0, min_dur])


def plt_final_tr_to_ph(tr_to_final_ph, index_th, ax, f_props, marker, tag):
    tr_to_final_ph = np.array(tr_to_final_ph)  # tr until final phase
    index_th = np.array(index_th)
    unq_ths = np.unique(index_th)
    for ind_th, th in enumerate(unq_ths):
        if th != -1 or tag == 'stages':   # only for those thresholds different than full task
            indx = index_th == th
            values_temp = tr_to_final_ph[indx]
            if len(values_temp) != 0:
                # plot number of trials
                ax.errorbar([th], np.nanmean(values_temp),
                            (np.std(values_temp)/np.sqrt(len(values_temp))),
                            color=f_props['color'], label=f_props['label'],
                            marker=marker, markersize=6)


def plt_tr_to_final_perf(tr_to_reach_perf, index_th, ax, f_props, marker):
    tr_to_reach_perf = np.array(tr_to_reach_perf)  # trials to reach final perf
    index_th = np.array(index_th)
    unq_ths = np.unique(index_th)
    for ind_th, th in enumerate(unq_ths):
        # for each threshold obtain corresponding trials
        indx = index_th == th
        values_temp = tr_to_reach_perf[indx]
        if len(values_temp) != 0:
            unq_ths_pos = unq_ths[unq_ths >= 0]
            unq_ths_pos.sort()
            x = min(unq_ths_pos)-(unq_ths_pos[1]-unq_ths_pos[0]) if ((th+1) < 0.01) else th
            # plot number of trials
            ax.errorbar([x], np.nanmean(values_temp),
                        (np.std(values_temp)/np.sqrt(len(values_temp))),
                        color=f_props['color'], label=f_props['label'],
                        marker=marker, markersize=6)


def plt_final_perf(final_perf, reached_ph, index_th, ax, f_props, marker):
    final_perf = np.array(final_perf)
    index_th = np.array(index_th)
    reached_ph = np.array(reached_ph)
    unq_ths = np.unique(index_th)
    for ind_th, th in enumerate(unq_ths):
        # only those traces with same threshold that have reached last phase
        indx = np.logical_and(index_th == th, reached_ph)
        assert len(indx) == len(index_th)
        values_temp = final_perf[indx]
        if len(values_temp) != 0:
            unq_ths_pos = unq_ths[unq_ths >= 0]
            unq_ths_pos.sort()
            x = min(unq_ths_pos)-(unq_ths_pos[1]-unq_ths_pos[0]) if ((th+1) < 0.01) else th
            # plot final perf
            ax.errorbar([x], np.nanmean(values_temp),
                        (np.std(values_temp)/np.sqrt(len(values_temp))),
                        color=f_props['color'], label=f_props['label'],
                        marker=marker, markersize=6)


def prop_of_exp_reaching_ph(reached_ph, index_th, ax, f_props, marker, tag):
    reached_ph = np.array(reached_ph)
    index_th = np.array(index_th)
    unq_ths = np.unique(index_th)
    for ind_th, th in enumerate(unq_ths):
        if th != -1 or tag == 'stages':  # for those thresholds different than full task
            indx = index_th == th
            # prop of traces that reached final phase
            prop = np.mean(reached_ph[indx])
            ax.plot(th, prop, color=f_props['color'], label=f_props['label'],
                    marker=marker, markersize=6)


def process_results_diff_thresholds(folder):
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    windows = ['0', '1', '2', '3', '4']  # , '500', '1000']
    markers = ['+', 'x', '1', 'o', '>']
    for alg in algs:
        print(alg)
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        ind = 0
        for ind_w, w in enumerate(windows):
            print('xxxxxxxxxxxxxxxxxxxxxxxx')
            print('Window')
            print(w)
            marker = markers[ind]
            ind += 1
            plot_results(folder, alg, w, limit_ax=False, marker=marker,
                         ax_final=ax, tag='th_stage',
                         f_final_prop={'color': clrs[ind_w], 'label': str(w)})

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0, 0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+'.png', dpi=200)
        plt.close(f)


def process_results_diff_protocols(folder):  # TODO: adapt for diff. ths results
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    windows = ['0', '1', '2', '3', '4']  # , '500', '1000']
    markers = ['+', 'x', '1', 'o', '>']
    for alg in algs:
        print(alg)
        f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        ind = 0
        for ind_w, w in enumerate(windows):
            print('xxxxxxxxxxxxxxxxxxxxxxxx')
            print('Window')
            print(w)
            marker = markers[ind]
            ind += 1
            plot_results(folder, alg, w, limit_ax=False, marker=marker,
                         ax_final=ax, tag='th_stage',
                         f_final_prop={'color': clrs[ind_w], 'label': str(w)})

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0, 0].legend(by_label.values(), by_label.keys())

        f.savefig(folder + '/final_results_' +
                  alg+'_'+'.png', dpi=200)
        plt.close(f)


if __name__ == '__main__':
    # folder = '/Users/martafradera/Desktop/OneDrive -' +\
    #          ' Universitat de Barcelona/TFG/bsc_stages/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/RL_algs/'
    # folder = '/home/manuel/CV-Learning/results/results_2303/one_agent_control/'
    folder = '/home/manuel/CV-Learning/results/results_2303/diff_protocols/'
    # folder = '/gpfs/projects/hcli64/shaping/one_agent_control/'
    # folder = '/gpfs/projects/hcli64/shaping/diff_protocols/'
    plt.close('all')
    process_results_diff_windows(folder)
