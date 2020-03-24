"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import glob
import gym
import seaborn as sns
import ntpath
from neurogym.envs import cv_learning


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
            for ind, value in enumerate(performance):
                if value == -1:
                    performance[ind] = 0
            ax.plot(steps, performance, 'k', label='Performance', ls='--')
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
                             legend=False, conv=[1]):
    data = put_together_files(folder)
    data_flag = True
    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(nrows=len(metrics.keys()), ncols=1,
                                 figsize=(8, 8))
        for ind_k, k in enumerate(metrics.keys()):
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
            if not ytitle:
                ax[ind_k].set_ylabel('mean ' + k)
            else:
                ax[ind_k].set_ylabel(ytitle)
            if legend:
                ax[ind_k].legend()
            if ind_k == len(metrics.keys())-1:
                ax[ind_k].set_xlabel('trials')
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
        #print(np.shape(curr_ph))
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


def plot_results(folder, algorithm, w,
                 keys=['performance', 'curr_ph'], limit_ax=True):
    clrs = sns.color_palette()
    files = glob.glob(folder + '*_' + algorithm + '_*_' + w)
    files += glob.glob(folder + algorithm + '*_full_*_')
    files = sorted(files)
    f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1, figsize=(8, 8))
    ths_mat = []
    ths_count = []
    th_index = []
    new_th = True
    fparams = {}
    metrics = {k: [] for k in keys}
    for ind_f, file in enumerate(files):
        f_name = ntpath.basename(file)
        th = f_name[f_name.find('th_stage')+9:]
        th = th[:th.find('_')]
        # check if th was already visited
        if th in ths_mat:
            ci = np.where(np.array(ths_mat) == th)[0][0]
            ths_count[ci] += 1
            new_th = False
        else:
            ths_mat.append(th)
            ths_count.append(1)
            ci = len(ths_mat)-1
            new_th = True
        metrics, flag = plot_rew_across_training(folder=file, ax=ax,
                                                 metrics=metrics,
                                                 conv=[1, 0],
                                                 fkwargs={'c': clrs[ci],
                                                          'lw': 0.5,
                                                          'alpha': 0.5})
        if flag:
            th_index.append(th)

        trials, fphase, fstart = find_reaching_phase_time(trace=metrics['curr_ph'],
                                                          phase=4)
        fperf = np.mean(metrics['performance'][0][-300:-1])
        if fphase == 4:
            perf = metrics['performance'][0]
            if len(perf) - fstart < 300:
                ph4_perf = np.mean(perf[fstart:-1])
            else:
                ph4_perf = np.mean(perf[-300:-1])
        else:
            ph4_perf = 'NaN'
        if new_th is False:
            fparams[str(th)] = {'final_perf': fparams[str(th)]['final_perf'].append(fperf),
                                'ph4_perf': fparams[str(th)]['ph4_perf'].append(ph4_perf),
                                'time': fparams[str(th)]['time'].append(trials)}
        else:
            fparams.update({str(th): {'final_perf': [fperf],
                                      'ph4_perf': [ph4_perf],
                                      'time': [trials]}})

    plot_fvalues(folder=folder, fparams=fparams, algorithm=algorithm, w=w)

    if metrics[keys[0]]:
        for ind_met, met in enumerate(metrics.keys()):
            plt_means(metric=metrics[met], index=th_index, ax=ax[ind_met],
                      clrs=clrs, limit_ax=limit_ax)
        np.array
        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[0].legend()
        f.savefig(folder+'/values_across_training_'+algorithm+'_'+w+'.png')
        # plot only means
        f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                             figsize=(8, 8))
        for ind_met, met in enumerate(metrics.keys()):
            plt_means(metric=metrics[met], index=th_index, ax=ax[ind_met],
                      clrs=clrs, limit_ax=limit_ax)

        ax[0].set_title(algorithm + ' (w: ' + w + ')')
        ax[0].legend()
        f.savefig(folder + '/mean_values_across_training_' +
                  algorithm+'_'+w+'.png')

        #fparams = np.array([['final_perf', fperf],
        #                    ['trials_until_phase4', trials],
        #                    ['phase4_perf', ph4_perf]])
        # np.savetxt(folder+'/final_mean_params_'+algorithm+'_'+w+'.txt',
        #           fparams, fmt='%s')
    else:
        plt.close(f)


def plot_fvalues(folder, fparams, algorithm, w):
    th_mat = []
    # f_perf = []
    ph4_perf = []
    time = []
    for th in fparams.keys():
        th_mat.append(float(th))
        # f_perf.append(np.mean(fparams[th]['final_perf']))
        ph4_perf.append(np.mean(fparams[th]['ph4_perf']))
        time.append(np.mean(fparams[th]['time']))

    f, ax = plt.subplots(sharex=True, nrows=2, ncols=1, figsize=(8, 8))
    ax[0].plot(th_mat, ph4_perf)
    ax[0].set_xlabel('Threshold')
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel('Performance')
    ax[0].set_title('Final performance in stage 4')
    ax[1].plot(th_mat, time)
    ax[1].set_xlabel('Threshold')
    ax[1].set_ylabel('Trials')
    ax[1].set_title('Time until reaching stage 4')
    f.suptitle(str(alg)+' (w: '+str(w)+')')

    f.savefig(folder+'/final_values_'+algorithm+'_'+w+'.png')


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
    unq_index = np.unique(index)
    for ind_th, th in enumerate(unq_index):
        indx = index == th
        traces_temp = metric[indx, :]
        ax.plot(np.nanmean(traces_temp, axis=0), color=clrs[ind_th],
                lw=1, label=th+'('+str(np.sum(indx))+')')
    if limit_ax:
        assert limit_mean, 'limiting ax only works when mean is also limited'
        ax.set_xlim([0, min_dur])


if __name__ == '__main__':
    folder = '/Users/martafradera/Desktop/OneDrive - Universitat de Barcelona/TFG/bsc_results/'
    plt.close('all')
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    windows = ['0', '2', '3']  # , '500', '1000']
    for alg in algs:
        print(alg)
        for w in windows:
            print(w)
            plot_results(folder, alg, w, limit_ax=False)
