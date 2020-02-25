"""Plotting functions."""

import numpy as np
import matplotlib.pyplot as plt
import glob
import gym
import seaborn as sns
import ntpath


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
        fig_(obs, actions, gt, rewards, legend=legend,
             states=states, name=name, obs_traces=obs_traces,
             fig_kwargs=fig_kwargs)
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
    obs = env.reset()  # TODO: not saving this first observation
    obs_cum_temp = obs
    for stp in range(int(num_steps_env)):
        if model is not None:
            action, _states = model.predict(obs)
            if isinstance(action, float) or isinstance(action, int):
                action = [action]
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
            perf.append(rew)
            obs_cum_temp = np.zeros_like(obs_cum_temp)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        if 'gt' in info.keys():
            gt.append(info['gt'])
        else:
            gt.append(0)
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    return observations, obs_cum, rewards, actions, perf,\
        actions_end_of_trial, gt, states


def fig_(obs, actions, gt=None, rewards=None, states=None, mean_perf=None,
         legend=True, obs_traces=[], name='', folder='', fig_kwargs={}):
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
    if len(obs.shape) != 2:
        raise ValueError('obs has to be 2-dimensional.')
    steps = np.arange(obs.shape[0])  # XXX: +1? 1st obs doesn't have action/gt

    n_row = 2  # observation and action
    n_row += rewards is not None
    n_row += states is not None

    gt_colors = 'gkmcry'
    if not fig_kwargs:
        fig_kwargs = dict(sharex=True, figsize=(5, n_row*1.5))

    f, axes = plt.subplots(n_row, 1, **fig_kwargs)
    # obs
    ax = axes[0]
    if len(obs_traces) > 0:
        assert len(obs_traces) == obs.shape[1],\
            'Please provide label for each trace in the observations'
        for ind_tr, tr in enumerate(obs_traces):
            ax.plot(obs[:, ind_tr], label=obs_traces[ind_tr])
        ax.legend()
        ax.set_xlim([-0.5, len(steps)-0.5])
    else:
        ax.imshow(obs.T, aspect='auto')
        ax.set_yticks([])

    if name:
        ax.set_title(name + ' env')
    ax.set_ylabel('Observations')

    # actions
    ax = axes[1]
    ax.plot(steps, actions, marker='+', label='Actions')
    if gt is not None:
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

    # rewards
    if rewards is not None:
        ax = axes[2]
        ax.plot(steps, rewards, 'r')
        ax.set_ylabel('Reward')
        if mean_perf is not None:
            ax.set_title('Mean performance: ' + str(np.round(mean_perf, 2)))
        ax.set_xlim([-0.5, len(steps)-0.5])

    # states
    if states is not None:
        ax.set_xticks([])
        ax = axes[3]
        plt.imshow(states[:, int(states.shape[1]/2):].T,
                   aspect='auto')
        ax.set_title('Activity')
        ax.set_ylabel('Neurons')

    ax.set_xlabel('Steps')
    plt.tight_layout()
    if folder is not None and folder != '':
        if folder.endswith('.png'):
            f.savefig(folder)
        else:
            f.savefig(folder + name + 'env_struct.png')
        plt.close(f)

    return f


def plot_rew_across_training(folder, window=1000, ax=None, ytitle='', xlbl='',
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


def plot_results(folder, algorithm, w,
                 keys=['reward', 'performance', 'curr_ph'], limit_ax=True):
    clrs = sns.color_palette()
    files = glob.glob(folder + '*_' + w + '_*_' + algorithm)
    files += glob.glob(folder + '*_full_*_' + algorithm)
    files = sorted(files)
    f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1, figsize=(8, 8))
    ths_mat = []
    ths_count = []
    th_index = []
    metrics = {k: [] for k in keys}
    for ind_f, file in enumerate(files):
        f_name = ntpath.basename(file)
        th = f_name[f_name.find('_')+1:]
        th = th[:th.find('_')]
        # check if th was already visited
        if th in ths_mat:
            ci = np.where(np.array(ths_mat) == th)[0][0]
            ths_count[ci] += 1
        else:
            ths_mat.append(th)
            ths_count.append(1)
            ci = len(ths_mat)-1
        metrics, flag = plot_rew_across_training(folder=file, ax=ax,
                                                 metrics=metrics,
                                                 conv=[1, 1, 0],
                                                 fkwargs={'c': clrs[ci],
                                                          'lw': 0.5,
                                                          'alpha': 0.5})
        if flag:
            th_index.append(th)
    for ind_met, met in enumerate(metrics.keys()):
        plt_means(metric=metrics[met], index=th_index, ax=ax[ind_met],
                  clrs=clrs, limit_ax=limit_ax)
    ax[0].set_title(alg + ' (w: ' + w + ')')
    ax[0].legend()
    f.savefig(folder + '/values_across_training_'+algorithm+'_'+w+'.png')


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
    # f = 'train_full_0_ACER'
    # plot_rew_across_training(folder=folder+f, fkwargs={'c': 'c'})
    plt.close('all')
    folder = '/home/molano/CV-Learning/results_1702/'
    algs = ['A2C', 'ACER', 'PPO2', 'ACKTR']
    windows = ['100']  # , '500', '1000']
    for alg in algs:
        print(alg)
        for w in windows:
            print(w)
            plot_results(folder, alg, w)
