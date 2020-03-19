#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:45:54 2020

@author: martafradera
"""


import numpy as np
import matplotlib.pyplot as plt
import glob
import gym
import seaborn as sns
import ntpath


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
            f.savefig(folder + '/mean_reward_across_training.svg')
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


def put_together_files(folder):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    print(files)
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


def plot_results(folder, algorithm, w, durs=False, stage2=False,
                 stage_change=False, keys=['reward', 'performance', 'curr_ph'],
                 limit_ax=True):

    files = glob.glob(folder + '*_' + w + '_*_' + algorithm)
    print(folder + '*_' + w + '_*_' + algorithm)
    print(files)
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
        if durs:
            fname = 'delays'
            metrics, flag = plot_durs(folder=file, ax=ax, metrics=metrics,
                                      fkwargs={'lw': 1, 'alpha': 1})
        elif stage2:
            fname = 'stage2'
            metrics, flag = plot_stage2(folder=file, ax=ax, metrics=metrics,
                                        fkwargs={'lw': 1, 'alpha': 1})
        elif stage_change:
            fname = 'stagechange'
            metrics, flag = plot_stage_change(folder=file, ax=ax,
                                              metrics=metrics,
                                              fkwargs={'lw': 1,
                                                       'alpha': 1})
        else:
            fname = 'values_across_training_'
            metrics, flag = plot_rew_across_training(folder=file, ax=ax,
                                                     metrics=metrics,
                                                     conv=[1, 1, 0],
                                                     fkwargs={'lw': 1,
                                                              'alpha': 1})
        if flag:
            th_index.append(th)

    ax[0].set_title(alg + ' (w: ' + w + ')')
    ax[0].legend()
    f.savefig(folder + '/' + fname + '_' + algorithm + '_' + w + '.svg')


def plot_durs(folder, window=0, ax=None, ytitle='', xlbl='',
              metrics={'inst_perf': [], 'inc_delays': [], 'days': [],
                       'durs': []},
              fkwargs={'c': 'tab:blue'}, legend=True, conv=[0]):

    data = put_together_files(folder)
    data_flag = True

    # find stage3 data
    ind = []
    for i, stage in enumerate(data['curr_ph']):
        if stage == 3:
            ind.append(i)

    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(nrows=len(metrics.keys()), ncols=1,
                                 figsize=(8, 8))
        for ind_k, k in enumerate(metrics.keys()):
            # days
            if k == 'days':
                trials_counter = data['trials_count'][ind[0]:ind[-1]]
                days = []
                day = 0
                for d in trials_counter:
                    if d == 0:
                        day += 1
                    days.append(day)
                metric = days
            else:
                metric = data[k][ind[0]:ind[-1]]
            if isinstance(window, float):
                if window < 1.0:
                    window = int(metric.size * window)
            if k == 'inst_perf':
                mean = np.convolve(metric, np.ones((window,))/window,
                                   mode='valid')
                up_th = np.repeat(data['th_perf'][0], len(mean))
                ax[ind_k].plot(up_th, label='upper threshold',
                               color='green', alpha=0.5)
                low_th = np.repeat(0.5, len(mean))
                ax[ind_k].plot(low_th, label='lower threshold',
                               color='red', alpha=0.5)
            else:
                mean = metric
            metrics[k].append(mean)

            if k == 'durs':
                ax[ind_k].plot(mean[:, 0], label='Delay short', **fkwargs)
                ax[ind_k].plot(mean[:, 1], label='Delay medium', **fkwargs)
                ax[ind_k].plot(mean[:, 2], label='Delay long', **fkwargs)

            else:
                ax[ind_k].plot(mean, **fkwargs)  # add color, label etc.

            ax[ind_k].set_xlabel(xlbl)
            ax[ind_k].set_ylabel(k)

            if legend:
                ax[ind_k].legend()
            if ind_k == len(metrics.keys())-1:
                ax[ind_k].set_xlabel('trials')
        if sv_fig:
            f.savefig(folder + '/delays.svg')
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


def plot_stage2(folder, window=10, ax=None, ytitle='', xlbl='',
                metrics={'inst_perf': [], 'curr_ph': []},
                fkwargs={'c': 'tab:blue'}, legend=False, conv=[0]):

    data = put_together_files(folder)
    data_flag = True

    # find stage1 to 3 data
    ind = []
    a = 0
    for i, stage in enumerate(data['curr_ph']):
        if 0 < stage < 3:
            ind.append(i)
        elif stage == 3 and a < 500:
            a += 1
            ind.append(i)

    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(nrows=len(metrics.keys()), ncols=1,
                                 figsize=(8, 8))
        for ind_k, k in enumerate(metrics.keys()):
            metric = data[k][ind[0]:ind[-1]]
            if isinstance(window, float):
                if window < 1.0:
                    window = int(metric.size * window)
            if k == 'inst_perf':
                mean = np.convolve(metric, np.ones((window,))/window,
                                   mode='valid')
                up_th = np.repeat(data['th_perf'][0], len(mean))
                ax[ind_k].plot(up_th, label='upper threshold',
                               color='green', alpha=0.5)
                low_th = np.repeat(0.5, len(mean))
                ax[ind_k].plot(low_th, label='lower threshold',
                               color='red', alpha=0.5)
            else:
                mean = metric
            metrics[k].append(mean)
            ax[ind_k].plot(mean, **fkwargs)  # add color, label etc.
            ax[ind_k].set_xlabel(xlbl)
            ax[ind_k].set_ylabel(k)
            if legend:
                ax[ind_k].legend()
            if ind_k == len(metrics.keys())-1:
                ax[ind_k].set_xlabel('trials')
        if sv_fig:
            f.savefig(folder + '/stage2.svg')
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


def plot_stage_change(folder, window=200, ax=None, ytitle='', xlbl='',
                      metrics={'curr_perf': [], 'curr_ph': [], 'days': []},
                      fkwargs={'c': 'tab:red'}, legend=True, conv=[0]):

    data = put_together_files(folder)
    data_flag = True
    clrs = sns.color_palette()

    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(nrows=len(metrics.keys()), ncols=1,
                                 figsize=(8, 8))
        for ind_k, k in enumerate(metrics.keys()):
            # days
            if k == 'days':
                trials_counter = data['trials_count']
                days = []
                day = 0
                for d in trials_counter:
                    if d == 0:
                        day += 1
                    days.append(day)
                metric = days
            elif k == 'curr_perf':
                metric = data[k][300:]
            else:
                metric = data[k]
            if isinstance(window, float):
                if window < 1.0:
                    window = int(metric.size * window)

            mean = metric
            metrics[k].append(mean)

            ax[ind_k].plot(mean, **fkwargs)
            ax[ind_k].set_ylabel(k)

            x = np.arange(len(mean))
            start = []
            stop = []
            control = 0
            prev_stage = -1
            first_day = []
            stages = []

            for ind, value in enumerate(mean):
                if k == 'curr_ph' or k == 'curr_perf':
                    a = data['keep_stage'][ind] * 1
                    if a == 1 and control == 0:
                        control = 1
                        start.append(ind)
                    elif a == 0 and control != 0:
                        stop.append(ind)
                        control = 0
                elif k == 'days':
                    if prev_stage != data['curr_ph'][ind]:
                        first_day.append(ind)
                        prev_stage = data['curr_ph'][ind]

            if k == 'curr_ph' or k == 'curr_perf':
                for ind, value in enumerate(start):
                    if value == start[0]:
                        ax[ind_k].plot(x[start[ind]:stop[ind]],
                                       mean[start[ind]:stop[ind]], color='red',
                                       **fkwargs, label='keep_stage')
                    else:
                        ax[ind_k].plot(x[start[ind]:stop[ind]],
                                       mean[start[ind]:stop[ind]], color='red',
                                       **fkwargs)

            if k == 'curr_perf':
                ax[ind_k].plot(data['th_perf'], label='threshold',
                               color='green', alpha=0.5)

            elif k == 'days':
                for ind, value in enumerate(first_day):
                    stage = data['curr_ph'][value]
                    label = 'stage ' + str(stage)
                    if value == first_day[-1]:
                        ax[ind_k].plot(x[first_day[ind]:len(mean)],
                                       mean[first_day[ind]:len(mean)],
                                       label=label, color=clrs[stage])

                    elif stage in stages:
                        ax[ind_k].plot(x[first_day[ind]:first_day[ind+1]-1],
                                       mean[first_day[ind]:first_day[ind+1]-1],
                                       color=clrs[stage])

                    else:
                        ax[ind_k].plot(x[first_day[ind]:first_day[ind+1]-1],
                                       mean[first_day[ind]:first_day[ind+1]-1],
                                       label=label, color=clrs[stage])
                    stages.append(stage)

            ax[ind_k].set_xlabel(xlbl)

            if legend:
                ax[ind_k].legend()
            if ind_k == len(metrics.keys())-1:
                ax[ind_k].set_xlabel('trials')
        if sv_fig:
            f.savefig(folder + '/change.svg')
    else:
        print('No data in: ', folder)
        data_flag = False

    return metrics, data_flag


if __name__ == '__main__':
    # f = 'train_full_0_ACER'
    # plot_rew_across_training(folder=folder+f, fkwargs={'c': 'c'})

    choices = ['stage_change', 'stage2', 'delays', 'rewards']
    figure = choices[0]

    plt.close('all')
    #folder = '/Users/martafradera/Desktop/OneDrive - Universitat de Barcelona/TFG/FIGURES/train_oldth/cas3/'
    folder = '/Users/martafradera/Desktop/OneDrive - Universitat de Barcelona/TFG/FIGURES/train/cas6/'
    algs = ['A2C']
    windows = ['300']  # , '500', '1000']

    for alg in algs:
        for w in windows:
            if figure == 'stage_change':
                plot_results(folder, alg, w, stage_change=True,
                             keys=['curr_perf', 'curr_ph', 'days'])
            elif figure == 'delays':
                plot_results(folder, alg, w, durs=True,
                             keys=['inst_perf', 'durs', 'days'])
            elif figure == 'stage2':
                plot_results(folder, alg, w, stage2=True,
                             keys=['inst_perf', 'curr_ph'])
            elif figure == 'rewards':
                plot_results(folder, alg, w)
                
