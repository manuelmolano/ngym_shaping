#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:19:22 2021

@author: leyreazcarate
"""


import numpy as np
import os
# import gym
import matplotlib.pyplot as plt
# import ngym_shaping as ng_sh
from ngym_shaping.utils import plotting as plot
import warnings
import itertools
from matplotlib import rcParams
import seaborn as sns
import glob
import ntpath
warnings.filterwarnings('default')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12

CLRS = sns.color_palette()

STAGES = [0, 1, 2, 3, 4]

PRTCLS_IND_MAP = {'01234': -1, '1234': 0, '0234': 1, '0134': 2, '0134X': 3,
                  '0124': 4, '034': 5, '234': 6, '34': 7, '4': 8}

THS_IND_MAP = {'full': 0.5, '0.6': 0.6, '0.65': 0.65, '0.7': 0.7,
               '0.75': 0.75, '0.8': 0.8, '0.85': 0.85, '0.9': 0.9}

# PUN_IND_MAP = {'0.0': 0, '-0.25': 1, '-0.5': 2}
PUN_IND_MAP = {'0.0': 0, '-0.25': 1, '-0.5': 2, '-0.75': 3, '-1.0': 4}

ALL_INDX = {}
ALL_INDX.update(PRTCLS_IND_MAP)
ALL_INDX.update(THS_IND_MAP)
ALL_INDX.update(PUN_IND_MAP)


### AUXILIAR FUNCTIONS TO LOAD DATA

def put_together_files(folder):
    """Put together all the files in a folder and organize them by sufix"""
    def order_by_sufix(file_list):
        sfx = [x[x.rfind('_')+1:x.rfind('.')].split('.') for x in file_list]
        stgs = np.array([x[0] for x in sfx])
        trs = np.array([x[1] for x in sfx])
        sorted_list = []
        for stg in np.unique(stgs):
            trs_tmp = np.array(trs[stgs == stg]).astype(float)
            f_lst_tmp = np.array(file_list)[stgs == stg]
            srtd_lst_tmp = [x for _, x in sorted(zip(trs_tmp, f_lst_tmp))]
            sorted_list += srtd_lst_tmp
        return sorted_list

    """Load all training data."""
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


def data_extraction(folder, metrics, w_conv_perf=500, conv=[1, 0]):
    """ Extract data saved during training.
    metrics: dict containing the keys of the data to loaextractd.
    conv: list of the indexes of the metrics to convolve."""
    # load all data from the same folder
    data = put_together_files(folder)
    data_flag = True
    if data:
        # extract each of the metrics
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


def learned(perf, learn_data, verbose=True, **params):
    """Do a smoothing (np.convolve) with a very long window.
    Make a histogram with the values of the resulting factor to see if
    You get 2 "mountains": chance and learned performance.
    Establish a threshold from that histogram.
    Find all values below / above thresholds
    Measure the minimum distance between the periods.
    """
    def get_event(trace, frst_lst):
        dwn_idx = np.where(np.diff(trace) < 0)[0]
        up_idx = np.where(np.diff(trace) > 0)[0]
        ev = None
        if frst_lst == 'first' and len(dwn_idx) > 0:
            ev = dwn_idx[0] if (dwn_idx[0] < up_idx).all() else None
        elif frst_lst == 'last' and len(up_idx) > 0:
            ev = up_idx[-1] if (up_idx[-1] > dwn_idx).all() else None
        return ev
    learn_dic_def = {'w_perf': 500, 'perf_bef_aft': [.6, .75]}
    learn_dic_def.update(params)
    w_perf = learn_dic_def['w_perf']
    perf_bef_aft = learn_dic_def['perf_bef_aft']
    perf_conv = np.convolve(perf, np.ones((w_perf,))/w_perf, mode='valid')

    not_learned = 1*(perf_conv < perf_bef_aft[0])
    ev_not_l = get_event(trace=not_learned, frst_lst='first')
    learned = 1*(perf_conv > perf_bef_aft[1])
    ev_l = get_event(trace=learned, frst_lst='last')
    if verbose:
        f, ax = plt.subplots(1, 1, figsize=(5,4))
        ax.plot(perf_conv, label='Performance')
        ax.plot([ev_not_l, ev_not_l], [0, 1], 'c', label='Start of learning period')
        ax.plot([ev_l, ev_l], [0, 1], 'm', label='End of learning period')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=perf_bef_aft[0], color='c', linestyle='--')
        ax.axhline(y=perf_bef_aft[1], color='m', linestyle='--')
        ax.set_ylim(0.4,1)
        ax.set_xlabel('Trials')
        ax.set_ylabel('Mean performance')
        ax.legend(loc='upper left')
    learned = False if (ev_l is None or ev_not_l is None or ev_l <= ev_not_l)\
        else True
    learn_data['learned'].append(learned)
    learn_data['ev_not_l'].append(ev_not_l)
    learn_data['ev_l'].append(ev_l)
    return learn_data
    # ax[1].hist(perf_conv, 50)
    # ax[1].plot([perf_bef_aft[0], perf_bef_aft[0]], [0, 100], 'c')
    # ax[1].plot([perf_bef_aft[1], perf_bef_aft[1]], [0, 100], 'm')
    # asd


def learning(folder, learn_data={}, verbose=True, conv=[1], **aha_dic):
    """ Extract data saved during training. metrics: dict containing
    the keys of the data to loaextractd.
    conv: list of the indexes of the metrics to convolve."""
    data = put_together_files(folder)  # load all data from the same folder
    data_flag = True
    if data:
        # extract each of the metrics
        if 'real_performance' in data.keys():
            perf = data['real_performance']
            stage = data['stage']
            perf = perf[stage == 1]
            learn_data = learned(perf=perf, learn_data=learn_data,
                                 **aha_dic)
    else:
        if verbose:
            print('No data in: ', folder)
        data_flag = False
    return learn_data, data_flag


def get_ahas(stage, perf, gt, aha_data, verbose=True, **aha_dic):
    """Find aha moments when all the requirements are fulfilled and
    plot them"""
    ahas_dic_def = {'w_ahas': 10, 'w_perf': 100,
                    'bef_aft_diff': 0.2, 'aha_th': 0.75, 'w_explore': 10}
    ahas_dic_def.update(aha_dic)
    prob_right = 0
    w_ahas = ahas_dic_def['w_ahas']
    w_perf = ahas_dic_def['w_perf']
    perf_th = ahas_dic_def['aha_th']
    bef_aft_diff = ahas_dic_def['bef_aft_diff']
    w_explore = ahas_dic_def['w_explore']
    no_shaping = len(np.unique(stage)) == 1 and 4 in stage
    if 1 in stage or no_shaping:
        indx = stage == 4 if no_shaping else stage == 1
        perf_stg_1 = perf[indx]
        gt = gt[indx]
        ahas = np.convolve(perf_stg_1, np.ones((w_ahas,))/w_ahas,
                           mode='valid')
        perf = np.convolve(perf_stg_1, np.ones((w_perf,))/w_perf,
                           mode='valid')
        if verbose:
            plt.figure(figsize=(4,3))
            # plt.title(folder)
            # plt.plot(perf_stg_1, '-+')
            plt.plot(ahas, '-+', label='Performance window = 10')
            plt.plot(perf, label='Performance window = 100')
            plt.legend()
            plt.xlabel('Trials')
            plt.ylabel('Mean performance')
            # plt.plot(np.convolve(perf_stg_1, np.ones((500,))/500,
            #                      mode='valid'))
        aha_indx = np.where(ahas > perf_th)[0]
        min_num_trs = 100
        aha_indx = aha_indx[aha_indx < len(perf_stg_1)-min_num_trs]
        aha_indx = aha_indx[aha_indx > min_num_trs]
        if len(aha_indx) > 0:
            prev_ai = -10e6
            for a_i in aha_indx:
                prev_perf = np.mean(perf_stg_1[a_i-w_perf:a_i])
                post_perf = np.mean(perf_stg_1[a_i+w_ahas:
                                               a_i+w_ahas+w_perf])
                aha_data['prev_prfs'].append(prev_perf)
                aha_data['post_prfs'].append(post_perf)
                # if verbose:
                #     plt.plot([a_i, a_i], [0, 1], '--m', lw=0.5)
                if prev_perf <= post_perf - bef_aft_diff and a_i > prev_ai+w_perf:
                    prev_ai = a_i
                    aha_data['aha_mmts'].append(a_i)
                    
                    if verbose:
                        plt.plot([a_i, a_i], [0, 1], '--k', lw=2)
                        print('AHA MOMENT')
                        print(gt[a_i-w_perf:a_i+w_ahas+w_perf])
                        print('**')
                    aha_data['gt_patterns'].append(gt[a_i-w_perf:
                                                      a_i+w_ahas+w_perf])
                    aha_data['perf_patterns'].append(perf_stg_1[a_i-w_perf:
                                                                a_i+w_ahas+w_perf])
                    # find probabilities of right before the aha window
                    right_number = np.sum(gt[a_i-w_explore:a_i] == 1)
                    prob_right = right_number/w_explore
                    aha_data['prob_right'].append(prob_right)
                    # find probabilities of right during the aha window
                    right_number = np.sum(gt[a_i:a_i+w_ahas] == 1)
                    prob_right = right_number/w_ahas
                    aha_data['prob_right_aha'].append(prob_right)

    return aha_data


def aha_moment(folder, aha_data={}, verbose=True, conv=[1], **aha_dic):
    """ Extract data saved during training. metrics: dict containing
    the keys of the data to loaextractd.
    conv: list of the indexes of the metrics to convolve."""
    data = put_together_files(folder)  # load all data from the same folder
    data_flag = True
    if data:
        # extract each of the metrics
        if 'real_performance' in data.keys():
            perf = data['real_performance']
            stage = data['stage']
            gt = data['gt']
            aha_data = get_ahas(stage=stage, perf=perf, gt=gt, aha_data=aha_data,
                                **aha_dic)
    else:
        if verbose:
            print('No data in: ', folder)
        data_flag = False
    return aha_data, data_flag


def get_tag(tag, file):
    """Process name"""
    f_name = ntpath.basename(file)
    assert f_name.find(tag) != -1, 'Tag '+tag+' not found in '+f_name
    val = f_name[f_name.find(tag)+len(tag)+1:]
    val = val[:val.find('_')] if '_' in val else val
    return val


def box_plot(data, ax, x, lw=.5, fliersize=4, color='k', widths=0.15):
    """Plotign box plots for different rollout values"""
    bp = ax.boxplot(data, positions=[x], widths=widths)
    for p in ['whiskers', 'caps', 'boxes', 'medians']:
        for bpp in bp[p]:
            bpp.set(color=color, linewidth=lw)
    bp['fliers'][0].set(markeredgecolor=color, markerfacecolor=color, 
                        alpha=0.5, marker='x', markersize=fliersize)
    rollout_names = ('5', '10', '20', '40')
    y_pos = np.arange(len(rollout_names))
    ax.set_xticks(y_pos)
    ax.set_xticklabels(rollout_names)
    ax.set_xlabel('Rollout')
    ax.set_ylabel('Performance')
    ax.set_title('Perfomance for different rollout values')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def get_hist(data, bins=None):
    """Create a histogram"""
    if bins is not None:
        hist, plt_bins = np.histogram(data, bins=bins)
    else:
        hist, plt_bins = np.histogram(data)
    hist = hist/np.sum(hist)
    plt_bins = plt_bins[:-1] + (plt_bins[1]-plt_bins[0])/2
    return hist, plt_bins


### FUNCTIONS TO OBTAIN VARIABLES
def tr_to_final_ph(stage, tr_to_ph, wind_final_perf, final_ph):
    """ Computes the number of trials required to reach the final phase.
    """
    time = np.where(stage == final_ph)[0]  # find those trials in phase 4
    reached = False
    if len(time) != 0:
        first_tr = np.min(time)  # min trial is first trial in phase 4
        if first_tr > len(stage) - wind_final_perf:
            # if phase 4 is not reached, last trial is obtained
            tr_to_ph.append(len(stage))
        else:
            tr_to_ph.append(first_tr)
            reached = True
    else:
        tr_to_ph.append(len(stage))
    return tr_to_ph, reached


def tr_to_reach_perf(perf, tr_to_ph, reach_perf, tr_to_perf, final_ph):
    """Computes the number of trials required to reach the final performance
    from those traces that do reach the final phase.
    """
    reached = False
    perf_in_final_ph = perf[tr_to_ph:]  # perf in the last phase
    time_above_th = np.where(perf_in_final_ph > reach_perf)[0]
    if len(time_above_th) == 0:
        tr_to_perf.append(len(perf))
    else:
        reached = True
        tr_to_perf.append(np.min(time_above_th) +
                          np.min(tr_to_ph))
    return tr_to_perf, reached


def compute_stability(perf, tr_ab_th):
    """Computes the performance stability after reaching the threshold
    performance, i.e. the proportion of instances with a performance greater
    than the chance level.
    """
    perf = np.array(perf)[tr_ab_th:]
    if perf.shape[0] != 0:
        forgetting_times = perf < 0.5
        stability = 1 - np.sum(forgetting_times)/perf.shape[0]
    else:
        stability = np.nan
    return stability


def get_noise(unq_vals):
    max_ = np.max([ALL_INDX[x] for x in unq_vals])
    min_ = np.min([ALL_INDX[x] for x in unq_vals])
    noise = (max_ - min_)/80
    return noise


### FUNCTIONS TO PLOT
def perf_hist(metric, ax, index, trials_day=300):
    """Plot a normalized histogram of the number of days/sessions spent with
    the same metric vale (e.g. performance).
    trials_day: number of trials to include on a session/day."""
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    bins = np.linspace(0, 1, 20)
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx]
        traces_temp = list(itertools.chain.from_iterable(traces_temp))
        hist_, plt_bins = np.histogram(traces_temp, bins=bins)
        hist_ = hist_/np.sum(hist_)
        plt_bins = plt_bins[:-1] + (plt_bins[1]-plt_bins[0])/2
        ax.plot(plt_bins, hist_, label=val, color=CLRS[ind_val])
    ax.legend()
    ax.set_xlabel('Performance')
    ax.set_ylabel('Days')


def plot_rew_across_training(metric, index, ax, n_traces=20,
                             selected_protocols=['-1.0', '-0.75', '-0.5',
                                                 '-0.25', '0.0']):
    """Plot traces across training, i.e. metric value per trial.
    """
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    for ind_val, val in enumerate(unq_vals):
        if val in selected_protocols:
            indx = index == val
            traces_temp = metric[indx][:n_traces]
            for trace in traces_temp:
                ax.plot(trace, color=CLRS[ind_val], alpha=0.5, lw=0.5)


def plt_means(metric, index, ax, limit_mean=True, limit_ax=True,
              selected_protocols=['-1.0', '-0.75', '-0.5', '-0.25', '0.0']):
    """Plot mean traces across training.
    """
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
        if val in selected_protocols:
            indx = index == val
            traces_temp = metric[indx, :]
            if not (np.isnan(traces_temp)).all():
                ax.plot(np.nanmean(traces_temp, axis=0), color=CLRS[ind_val],
                        lw=1, label=val+' ('+str(np.sum(indx))+')')
    if limit_ax:
        assert limit_mean, 'limiting ax only works when mean is also limited'
        ax.set_xlim([0, min_dur])


def plt_perf_indicators(values, index_val, ax, f_props, ax_props, reached=None,
                        discard=[], plot_individual_values=True,
                        errorbars=True):
    """Plot final results, in this case, performance indicators
    """
    values = np.array(values)
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
                ax.plot(xs, values_temp, alpha=0.5, linestyle='None',
                        **f_props)
    ax.set_xlabel(ax_props['tag'])
    ax.set_ylabel(ax_props['ylabel'])
    ax.set_xticks(ax_props['ticks'])
    ax.set_xticklabels(ax_props['labels'])


def trials_per_stage(metric, ax, index):
    """Plot the mean number of trials spent on each of the stages."""
    bins = np.linspace(STAGES[0]-0.5, STAGES[-1]+.5, len(STAGES)+1)
    metric = np.array(metric)
    index = np.array(index)
    unq_vals = np.unique(index)
    # find all the trials spent on each stage
    for ind_val, val in enumerate(unq_vals):
        indx = index == val
        traces_temp = metric[indx]
        counts_mat = []
        n_traces = len(traces_temp)
        for ind_tr in range(n_traces):
            # plot the individual values
            counts = np.histogram(traces_temp[ind_tr], bins=bins)[0]
            indx = counts != 0
            noise = np.random.normal(0, 0.01, np.sum(indx))
            ax.plot(np.array(STAGES)[indx]+noise, counts[indx], '+',
                    color=CLRS[ind_val], alpha=0.5)
            counts_mat.append(counts)
        counts_mat = np.array(counts_mat)
        mean_counts = np.mean(counts_mat, axis=0)
        # (e.g. in protocol 0234, don't plot stage 1)
        # ax.errorbar(np.array(STAGES), mean_counts, std_counts, marker='+',
        #             color=CLRS[ind_val], label=val)
        # std_counts = np.std(counts_mat, axis=0)/np.sqrt(n_traces)
        indx = mean_counts != 0
        # plot the mean values
        ax.plot(np.array(STAGES)[indx], mean_counts[indx], marker='+',
                linestyle='--', color=CLRS[ind_val], label=val)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.yscale('log')
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Stage')
    ax.set_ylabel('Trials')


def plot_inst_punishment(num_instances, punish_3_vector, conv_w):
    """plot of each punishment and each instance"""
    for i_i in range(num_instances):
        for pun in punish_3_vector:
            sv_f_inst = sv_f+'/pun_'+str(round(pun, 2))+'_inst_'+str(i_i)+'/'
            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].set_title('Punishment' + str(round(pun, 2)))
            # plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
            #                               fkwargs={'c': 'tab:red'},
            #                               legend=False, zline=False,
            #                               metric_name='performance',
            #                               window=conv_w)
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                          fkwargs={'c': 'tab:blue'},
                                          legend=False, zline=False,
                                          metric_name='real_performance',
                                          window=conv_w)
            ax[0].axhline(y=TH, linestyle='--', color='k')
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                          fkwargs={'c': 'tab:blue'},
                                          legend=False, zline=False,
                                          metric_name='stage',
                                          window=conv_w)
            f.savefig(sv_f_inst+'.png', dpi=300)


def plot_results(folder, setup='', setup_nm='', w_conv_perf=500, perf_th=0.6,
                 keys=['real_performance', 'stage'], limit_ax=True, final_ph=4,
                 ax_final=None, tag='th_stage', limit_tr=False, rerun=False,
                 f_final_prop={'color': (0, 0, 0), 'label': '', 'marker': '.'},
                 plt_ind_vals=True, plt_ind_traces=True, n_roll=5, name='',
                 x=0, ahas_dic={}, learn_dic={}):
    """This function uses the data generated during training to analyze it
    and generate figures showing the results in function of the different
    values used for the third level variable (i.e. differen threshold values
    or different shaping protocols).
    folder: folder where we store/load the data.
    algorithm: used algorithm for training.
    setup: value indicating the second level variable value, i.e. the used
    number of window or the used n_ch (number of channels) for training.
    setup_nm: indicates which second level exploration has been done
    (window/n_ch).
    w_conv_perf: dimension of the convolution window.
    keys: list of the names of the metrics to explore.
    limit_ax: limit axis when plottingg.
    final_ph: stage number that corresponds to the last stage of training.
    perf_th: threshold performance to separate the traces that have or have not
    learnt the task.
    ax_final: axes for plotting the final results.
    tag: name of the performed exploration ('th_stage' for different threshold
    values, and 'stages' for different shaping protocols').
    limit_tr: limit trace when plotting.
    rerun: regenerating the data obtained from the metrics during training.
    f_final_prop: plotting kwargs.
    plt_ind_vals: include the individual values (results for each trace) in the
    final plot.
    plt_ind_traces: plot traces across training.
    """
    # assert ('performance' in keys) and ('stage' in keys),\
    #     'performance and stage need to be included in the metrics (keys)'
    # PROCESS RAW DATA
    if not os.path.exists(folder+'/data'+'_'+setup_nm+'_'+setup +
                          '.npz') or rerun:
        print('Pre-processing raw data')
        files = glob.glob(folder+'*'+setup_nm+'_'+setup+'*')
        assert len(files) > 0, 'No files of the form: ' + folder + '*' +\
            setup_nm + '_'+setup+'_*'
        # files = sorted(files)
        val_index = []  # stores values for each instance
        metrics = {k: [] for k in keys}
        aha_data = {'aha_mmts': [], 'prev_prfs': [], 'post_prfs': [],
                    'gt_patterns': [], 'perf_patterns': [], 'prob_right': [],
                    'prob_right_aha': []}
        learn_data = {'learned': [], 'ev_not_l': [], 'ev_l': []}
        keys = np.array(keys)
        for ind_f, file in enumerate(files):
            print(file)
            val = get_tag(tag, file)
            # get metrics
            metrics, flag = data_extraction(folder=file, metrics=metrics,
                                            w_conv_perf=w_conv_perf,
                                            conv=[1, 0])
            aha_data, flag = aha_moment(folder=file, aha_data=aha_data,
                                        **ahas_dic)
            learn_data, flag = learning(folder=file, learn_data=learn_data,
                                        **learn_dic)

            # store values
            if flag:
                val_index.append(val)
        val_index = np.array(val_index)
        # AHA-MOMENT
        aha_mmts = aha_data['aha_mmts']
        prev_prfs = aha_data['prev_prfs']
        post_prfs = aha_data['post_prfs']
        gt_patterns = aha_data['gt_patterns']
        perf_patterns = aha_data['perf_patterns']
        prob_right = aha_data['prob_right']
        prob_right_aha = aha_data['prob_right_aha']
        if len(aha_mmts) > 0:
            fig, ax1 = plt.subplots()
            colors = ['b', 'g']
            labels = ['prev_prfs', 'post_prfs']
            ax1.hist([prev_prfs, post_prfs], bins=10, color=colors, label=labels)
            ax1.legend()
            plt.tight_layout()
            plt.show()

        names = ['values_across_training_']  # 'mean_values_across_training_']
        ylabels = ['Performance', 'Phase', 'Number of steps',
                   'Session performance']
        ax_final_perfs = ax_final[1]
        metrics['real_performance']
        final_wind = 100
        final_perfs = [np.mean(p[-final_wind:])
                       for p in metrics['real_performance']]
        box_plot(data=final_perfs, ax=ax_final_perfs, x=x)
        num_sh = 100000
        bins = np.linspace(0, 1, 10)
        # probabilities of right
        w = 10
        r_m = np.random.rand(num_sh, w)
        r_m = np.sum(r_m > 0.5, axis=1)/w
        prob_R_chance, plt_bins = get_hist(r_m, bins=bins)
        f, ax = plt.subplots(1, 1)
        ax.plot(plt_bins, prob_R_chance)
        prob_R, plt_bins = get_hist(prob_right, bins=bins)
        ax.plot(plt_bins, prob_R)
        ax.legend(labels=('Ground truth', 'Right side trials'))
        ax.set_title('Probabilities of ground truth=right before aha-moment')
        # probabilities of right aha
        w = 10
        r_m = np.random.rand(num_sh, w)
        r_m = np.sum(r_m > 0.5, axis=1)/w
        prob_R_chance, plt_bins = get_hist(r_m, bins=bins)
        f, ax = plt.subplots(1, 1)
        ax.plot(plt_bins, prob_R_chance)
        prob_R, plt_bins = get_hist(prob_right_aha, bins=bins)
        ax.plot(plt_bins, prob_R)
        ax.legend(labels=('Ground truth', 'Right side trials'))
        ax.set_title('Probabilities of ground truth=right during aha-moment')

        # number of aha moments for each subj
        subj_length = [len(aha_mmts)]
        f, ax = plt.subplots(1, 1, figsize=(4.5, 5))
        ax.hist(subj_length, bins=np.arange(6)-0.5)
        ax.set_xlabel('Number of aha moments')
        ax.set_ylabel('Number of subjects')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        print('Mean/std number of aha-moments')
        print(np.mean(subj_length))
        print(np.std(subj_length))
        ax.set_title('Number of aha moments for each subject')
        f, ax = plt.subplots(1, 2)
        learned_mat = 1*np.array(learn_data['learned'])
        ax[0].hist(learned_mat)
        ax[0].set_title('Subjects that learn (1 learn, 0 not)')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        zip_ = zip(learn_data['ev_l'], learn_data['ev_not_l'])
        learning_time = [x-y for x, y in zip_ if x is not None and y is not None]
        ax[1].hist(learning_time, 8)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].set_xlabel('Time to learn')
        ax[1].set_ylabel('Counts')
        np.mean(learning_time)
        np.std(learning_time)
        min(learning_time)

        for ind in range(len(names)):
            f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                                 figsize=(12, 12))
            # plot means
            for ind_met, met in enumerate(keys):
                metric = metrics[met]
                if plt_ind_traces:
                    plot_rew_across_training(metric=metric, index=val_index,
                                             ax=ax[ind_met], n_traces=3)
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], limit_ax=limit_ax)
                ax[ind_met].set_ylabel(ylabels[ind_met])
            ax[0].set_title('Roll_out = ' + str(n_roll))
            ax[0].axhline(y=0.55, linestyle='--', color='k')
            ax[0].set_xlabel('Trials')
            ax[len(keys)-1].set_xlabel('Trials')
            ax[len(keys)-1].legend()
            f.savefig(folder+'/'+names[ind]+'_'+setup_nm+'_'+setup +
                      '_'+str(limit_tr)+'.png', dpi=200)
            # plt.close(f)

        # plot days under perf
        if 'curr_perf' in keys:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
            metric = metrics['curr_perf']
            perf_hist(metric, ax=ax, index=val_index, trials_day=300)
            ax.set_title('Performance histogram ('+')')
            f.savefig(folder+'/perf_hist_'+'_'+setup_nm+'_'+setup +
                      '.svg', dpi=200)
            # plt.close(f)

        # plot trials per stage
        if 'stage' in keys:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            metric = metrics['stage']
            trials_per_stage(metric, ax=ax, index=val_index)
            ax.set_title('Average number of trials per stage ('+')')
            f.savefig(folder+'/trials_stage_'+'_'+setup_nm+'_' +
                      setup+'.svg', dpi=200)
            plt.close(f)

        # PROCESS TRACES AND SAVE DATA
        tr_to_perf = []  # stores trials to reach final performance
        reached_ph = []  # stores whether the final phase is reached
        reached_perf = []  # stores whether the pre-defined perf is reached
        exp_durations = []  # stores the total number of explored trials
        stability_mat = []  # stores the performance stability
        final_perf = []  # stores the average final performance
        tr_to_ph = []  # stores trials to reach final phase
        stps_to_perf = []  # stores steps to final performance
        stps_to_ph = []  # stores steps to final performance
        if limit_tr:
            min_dur = np.min([len(x) for x in metrics['stage']])
        else:
            min_dur = np.max([len(x) for x in metrics['stage']])

        for ind_f in range(len(metrics['stage'])):
            # store durations
            exp_durations.append(len(metrics['stage'][ind_f]))
            for k in metrics.keys():
                metrics[k][ind_f] = metrics[k][ind_f][:min_dur]
                if len(metrics[k][ind_f]) == 0:
                    metrics[k][ind_f] = np.nan*np.ones((min_dur,))
            # phase analysis
            stage = metrics['stage'][ind_f]
            # number of trials until final phase
            tr_to_ph, reached = tr_to_final_ph(stage, tr_to_ph, w_conv_perf,
                                               final_ph)
            reached_ph.append(reached)
            # performance analysis
            perf = np.array(metrics['real_performance'][ind_f])
            # get final performance
            final_perf.append(perf[-1])
            # get trials to reach specified performance
            tt_ph = tr_to_ph[-1]
            tr_to_perf, reached = tr_to_reach_perf(perf=perf.copy(),
                                                   tr_to_ph=tt_ph,
                                                   reach_perf=perf_th,
                                                   tr_to_perf=tr_to_perf,
                                                   final_ph=final_ph)
            reached_perf.append(reached)
            # performance stability
            tt_prf = tr_to_perf[-1]
            stability_mat.append(compute_stability(perf=perf.copy(),
                                                   tr_ab_th=tt_prf))
        data = {'tr_to_perf': tr_to_perf, 'reached_ph': reached_ph,
                'reached_perf': reached_perf, 'exp_durations': exp_durations,
                'stability_mat': stability_mat, 'final_perf': final_perf,
                'tr_to_ph': tr_to_ph, 'stps_to_perf': stps_to_perf,
                'stps_to_ph': stps_to_ph, 'val_index': val_index}
        np.savez(folder+'/data'+'_'+setup_nm+'_'+setup+'.npz',
                 **data)
    # LOAD AND (POST)PROCESS DATA
    print('Loading data from: ', folder+'/data'+'_'+setup_nm +
          '_'+setup+'.npz')
    tmp = np.load(folder+'/data'+'_'+setup_nm+'_'+setup+'.npz',
                  allow_pickle=True)
    # the loaded file does not allow to modifying it
    data = {}
    for k in tmp.keys():
        data[k] = list(tmp[k])
    val_index = data['val_index']
    print('Plotting results')
    # define xticks
    ax_props = {'tag': tag}
    if tag == 'stages':
        ax_props['labels'] = list(PRTCLS_IND_MAP.keys())
        ax_props['ticks'] = list(PRTCLS_IND_MAP.values())
    elif tag == 'th_stage':
        ax_props['labels'] = list(THS_IND_MAP.keys())
        ax_props['ticks'] = list(THS_IND_MAP.values())
    elif tag == 'pun':
        ax_props['labels'] = list(PUN_IND_MAP.keys())
        ax_props['ticks'] = list(PUN_IND_MAP.values())

    # plot results
    ax1 = ax_final[0]
    ax3 = ax_final[2]
    # final figures
    # prop of instances reaching phase 4
    ax_props['ylabel'] = 'Proportion of instances reaching phase ' + \
        str(final_ph)
    plt_perf_indicators(values=data['reached_ph'], index_val=val_index,
                        ax=ax1[0], f_props=f_final_prop,
                        ax_props=ax_props, discard=['full', '4'],
                        errorbars=False, plot_individual_values=False)
    # trials to reach phase 4
    ax_props['ylabel'] = 'Number of trials to reach phase '+str(final_ph)
    plt_perf_indicators(values=data['tr_to_ph'],
                        f_props=f_final_prop, ax_props=ax_props,
                        index_val=val_index, ax=ax1[1],
                        reached=data['reached_ph'], discard=['full', '4'],
                        plot_individual_values=plt_ind_vals)
    handles, labels = ax1[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1[0].legend(by_label.values(), by_label.keys())
    ax1[1].set_yscale('log')

    # plot final performance
    ax_props['ylabel'] = 'Average performance'
    plt_perf_indicators(values=data['final_perf'],
                        reached=data['reached_ph'],
                        f_props=f_final_prop, index_val=val_index,
                        ax=ax3[0, 0], ax_props=ax_props,
                        plot_individual_values=plt_ind_vals)
    # prop of trials that reach final perf
    ax_props['ylabel'] = 'Proportion of instances reaching final perf'
    plt_perf_indicators(values=data['reached_perf'], index_val=val_index,
                        ax=ax3[0, 1], f_props=f_final_prop,
                        reached=data['reached_ph'], ax_props=ax_props,
                        errorbars=False, plot_individual_values=False)
    # trials to reach final perf
    ax_props['ylabel'] = 'Number of trials to reach final performance'
    plt_perf_indicators(values=data['tr_to_perf'],
                        reached=data['reached_perf'],
                        index_val=val_index, ax=ax3[1, 0],
                        f_props=f_final_prop, ax_props=ax_props,
                        plot_individual_values=plt_ind_vals)
    ax3[1, 0].set_yscale('log')
    # plot stability
    ax_props['ylabel'] = 'Stability'
    plt_perf_indicators(values=data['stability_mat'], index_val=val_index,
                        ax=ax3[1, 1], f_props=f_final_prop,
                        ax_props=ax_props, reached=data['reached_perf'],
                        plot_individual_values=plt_ind_vals)
    handles, labels = ax3[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3[0, 0].legend(by_label.values(), by_label.keys())


### MAIN

if __name__ == '__main__':
    plt.close('all')
    # sv_f = '/home/molano/shaping/results_280421/no_shaping/'
    # sv_f = '/home/manuel/shaping/results_280421/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/results_280421/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/no_shaping/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'shaping_long_tr_one_agent/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'no_shaping_long_tr_one_agent/'
    # sv_f = '/home/molano/shaping/results_280421/' +\
    #     'shaping_long_tr_one_agent/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'no_shaping_long_tr_one_agent_stg_4/'
    # sv_f = '/home/molano/shaping/results_280421/' +\
    #     'no_shaping_long_tr_one_agent/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'shaping_diff_punishment/'
    # sv_f = '/home/manuel/shaping/results_280421/shaping_long_tr_one_agent/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'no_shaping_long_tr_one_agent_stg_4_nsteps_40/'
    # sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/' +\
    #     'no_shaping_long_tr_one_agent_stg_4_nsteps_20/'
    sv_f = '/Users/leyreazcarate/Desktop/TFG/results_280421/shaping_5_0.1/'
    # sv_f = '/home/manuel/shaping/results_280421/shaping_5_0.1/'
    # sv_f = '/home/molano/shaping/results_280421/shaping_5_0.1/'
    NUM_STEPS = 200000  # 1e5*np.arange(10, 21, 2)
    TH = 0.6
    # TODO: tune perf_bef_aft, bef_aft_diff
    ahas_dic = {'w_ahas': 10, 'w_perf': 1000,
                'bef_aft_diff': 0.1, 'aha_th': 0.65, 'w_explore': 10}

    learn_dic = {'w_perf': 500, 'perf_bef_aft': [.55, .6]}

    plot_separate_figures = True
    plot_all_figs = True
    num_instances = 3
    stages = np.arange(5)
    perf_w = 100
    stg_w = 1000
    conv_w = 50
    final_ph = 4
    # f1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # f2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # f3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    # ax = [ax1, ax2, ax3]
    # plot_results(folder=sv_f, setup_nm='pun', w_conv_perf=perf_w,
    #              keys=['real_performance', 'stage'], limit_ax=True,
    #              final_ph=final_ph, ax_final=ax, perf_th=TH,
    #              tag='pun', limit_tr=False, rerun=True,
    #              f_final_prop={'color': (0, 0, 0), 'label': '', 'marker': '.'},
    #              plt_ind_vals=True, plt_ind_traces=True, ahas_dic=ahas_dic,
    #              learn_dic=learn_dic)

    # f1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # f3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    # f2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # ax = [ax1, ax2, ax3]
    # plot_results(folder=sv_f, setup_nm='pun', w_conv_perf=perf_w,
    #               keys=['real_performance', 'stage'], limit_ax=True,
    #               final_ph=final_ph, ax_final=ax, tag='pun', limit_tr=False,
    #               rerun=True, plt_ind_vals=True, plt_ind_traces=True,
    #               f_final_prop={'color': (0, 0, 0), 'label': '', 'marker': '.'},
    #               **ahas_dic)

    # PLOT FIGURES NO-SHAPING DIFFERENT ROLLOUTS
    main_folder = '/Users/leyreazcarate/Desktop/TFG/results_280421/'
    # main_folder = '/home/manuel/shaping/results_280421/'
    rollouts = [5, 10, 20, 40]
    f2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
    for i_ro, ro in enumerate(rollouts):
        sv_f = main_folder+'no_shaping_long_tr_one_agent_stg_4_nsteps_' + \
            str(ro)+'/'

        f1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        f3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        ax = [ax1, ax2, ax3]
        plot_results(folder=sv_f, setup_nm='pun', w_conv_perf=perf_w,
                      keys=['real_performance', 'stage'], limit_ax=True,
                      final_ph=final_ph, ax_final=ax, x=i_ro,
                      tag='pun', limit_tr=False, rerun=True,
                      f_final_prop={'color': (0, 0, 0), 'label': '',
                                    'marker': '.'},
                      plt_ind_vals=True, plt_ind_traces=True, n_roll=ro,
                      ahas_dic=ahas_dic, learn_dic=learn_dic)
        f1.savefig(sv_f + '/final_results_phase.svg', dpi=200)
        f3.savefig(sv_f + '/final_results_performance.svg', dpi=200)
        plt.close(f1)
        plt.close(f3)
    f2.savefig(main_folder + '/final_results_steps.svg', dpi=200)

