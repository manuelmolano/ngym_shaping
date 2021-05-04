#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 10:19:22 2021

@author: leyreazcarate
"""


import numpy as np
import os
import gym
import matplotlib.pyplot as plt
from ngym_shaping.utils import plotting as plot
import warnings
from matplotlib import rcParams
import seaborn as sns
import glob
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

ALL_INDX = {}
ALL_INDX.update(PRTCLS_IND_MAP)
ALL_INDX.update(THS_IND_MAP)


# plot of each punishment and each instance
def learning(num_instances, punish_3_vector, sv_f, stages, perf_w, stg_w,
             env_kwargs):
    for i_i in range(num_instances):
        for pun in punish_3_vector:
            sv_f_inst = sv_f+'/pun_'+str(round(pun, 2))+'_inst_'+str(i_i)+'/'
            print('---------')
            print(sv_f_inst)
            print('---------')
            if not os.path.exists(sv_f_inst+'/bhvr_data_all.npz') or RERUN:
                rewards = {'abort': -0.1, 'correct': +1., 'fail': pun}
                env_kwargs['rewards'] = rewards
                env = ng_sh.envs.DR_stage.shaping(stages=stages, th=TH,
                                                  perf_w=perf_w,
                                                  stg_w=stg_w,
                                                  sv_folder=sv_f_inst,
                                                  sv_per=stg_w, **env_kwargs)
                if LEARN:
                    env = DummyVecEnv([lambda: env])
                    # Define model
                    model = A2C(LstmPolicy, env, verbose=1,
                                policy_kwargs={'feature_extraction': "mlp"})
                    # Train model
                    model.learn(total_timesteps=NUM_STEPS, log_interval=10e10)
                    model.save(sv_f_inst+'model')
                else:
                    env.reset()
                    for ind in range(NUM_RAND):
                        if np.random.rand() < (rand_act_prob-2*pun):
                            action = np.random.randint(0, 3)
                        else:
                            action = env.gt_now  # correct action (groundtruth)
                        env.step(action)
                env.close()


def plot_inst_punishment(num_instances, punish_3_vector, conv_w):
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


def plot_figs(punish_6_vector, num_instances, conv_w):
    f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fkwargs = {'c': 'tab:blue'}
    for i_p, pun in enumerate(punish_6_vector):
        for i_i in range(num_instances):
            sv_f_inst = sv_f+'/pun_'+str(round(pun, 2))+'_inst_'+str(i_i)+'/'
            pun_str = str(round(pun, 2))
            sv_f_inst = sv_f+'/pun_'+pun_str+'_inst_'+str(i_i)+'/'
            fkwargs['alpha'] = 1-1/(i_p+2)
            fkwargs['label'] = 'pun = '+pun_str if i_i == 0 else ''
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                          fkwargs=fkwargs, legend=False,
                                          zline=False, window=conv_w,
                                          metric_name='real_performance')
            ax[0].axhline(y=TH, linestyle='--', color='k')
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                          fkwargs=fkwargs, legend=False,
                                          zline=False, window=conv_w,
                                          metric_name='stage')
    f.tight_layout()
    ax[0].legend()
    f.savefig(sv_f+'all_insts.png', dpi=300)


def plot_results(folder, algorithm, setup='', setup_nm='', w_conv_perf=500,
                 keys=['performance', 'curr_ph', 'num_stps', 'curr_perf'],
                 limit_ax=True, final_ph=4, perf_th=0.7, ax_final=None,
                 tag='th_stage', limit_tr=False, rerun=False,
                 f_final_prop={'color': (0, 0, 0), 'label': ''},
                 plt_ind_vals=True, plt_ind_traces=True):
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
    assert ('performance' in keys) and ('curr_ph' in keys),\
        'performance and curr_ph need to be included in the metrics (keys)'
    # PROCESS RAW DATA
    if not os.path.exists(folder+'/data'+algorithm+'_'+setup_nm+'_'+setup +
                          '.npz') or rerun:
        print('Pre-processing raw data')
        files = glob.glob(folder+'alg_'+algorithm+'*'+setup_nm+'_'+setup+'_*')
        assert len(files) > 0, 'No files of the form: '+folder+'*'+algorithm +\
            '*'+setup_nm+'_'+setup+'_*'
        files = sorted(files)
        val_index = []  # stores values for each instance
        metrics = {k: [] for k in keys}
        keys = np.array(keys)
        for ind_f, file in enumerate(files):
            print(file)
            val = get_tag(tag, file)
            # get metrics
            metrics, flag = data_extraction(folder=file, metrics=metrics,
                                            w_conv_perf=w_conv_perf,
                                            conv=[1, 0, 1, 0])
            # store values
            if flag:
                val_index.append(val)
        val_index = np.array(val_index)

        # np.savez(folder+'/metrics'+algorithm+'_'+setup_nm+'_'+setup+'.npz',
        #          **metrics)

        # LOAD AND (POST)PROCESS DATA
        # print('Loading data from: ', folder+'/metrics'+algorithm+'_'+setup_nm
        #       +'_'+setup+'.npz')
        # tmp = np.load(folder+'/metrics'+algorithm+'_'+setup_nm+'_'+setup
        #               +'.npz', allow_pickle=True)
        # the loaded file does not allow to modifying it
        # metrics = {}
        # for k in tmp.keys():
        #     metrics[k] = list(tmp[k])

        names = ['values_across_training_', 'mean_values_across_training_']
        ylabels = ['Performance', 'Phase', 'Number of steps',
                   'Session performance']
        for ind in range(2):
            f, ax = plt.subplots(sharex=True, nrows=len(keys), ncols=1,
                                 figsize=(12, 12))
            # plot means
            for ind_met, met in enumerate(keys):
                metric = metrics[met]
                if ind == 0 and plt_ind_traces:
                    plot_rew_across_training(metric=metric, index=val_index,
                                             ax=ax[ind_met])
                plt_means(metric=metric, index=val_index,
                          ax=ax[ind_met], limit_ax=limit_ax)
                ax[ind_met].set_ylabel(ylabels[ind_met])
            ax[0].set_title(algorithm + ' ('+setup_nm+': ' + setup + ')')
            ax[len(keys)-1].set_xlabel('Trials')
            ax[len(keys)-1].legend()
            f.savefig(folder+'/'+names[ind]+algorithm+'_'+setup_nm+'_'+setup +
                      '_'+str(limit_tr)+'.svg', dpi=200)
            plt.close(f)

        # plot days under perf
        if 'curr_perf' in keys:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
            metric = metrics['curr_perf']
            perf_hist(metric, ax=ax, index=val_index, trials_day=300)
            ax.set_title('Performance histogram ('+algorithm+')')
            f.savefig(folder+'/perf_hist_'+algorithm+'_'+setup_nm+'_'+setup +
                      '.svg', dpi=200)
            plt.close(f)

        # plot trials per stage
        if 'curr_ph' in keys:
            f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
            metric = metrics['curr_ph']
            trials_per_stage(metric, ax=ax, index=val_index)
            ax.set_title('Average number of trials per stage ('+algorithm+')')
            f.savefig(folder+'/trials_stage_'+algorithm+'_'+setup_nm+'_' +
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
            min_dur = np.min([len(x) for x in metrics['curr_ph']])
        else:
            min_dur = np.max([len(x) for x in metrics['curr_ph']])

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
            # # number of steps
            if len(metrics['num_stps'][ind_f]) != 0:
                num_steps = np.cumsum(metrics['num_stps'][ind_f])
                stps_to_perf.append(num_steps[max(0, tt_prf-1)]/1000)
                stps_to_ph.append(num_steps[min(max(0, tt_ph-1),
                                                len(num_steps)-1)]/1000)
            else:
                stps_to_perf.append(np.nan)
                stps_to_ph.append(np.nan)
        data = {'tr_to_perf': tr_to_perf, 'reached_ph': reached_ph,
                'reached_perf': reached_perf, 'exp_durations': exp_durations,
                'stability_mat': stability_mat, 'final_perf': final_perf,
                'tr_to_ph': tr_to_ph, 'stps_to_perf': stps_to_perf,
                'stps_to_ph': stps_to_ph, 'val_index': val_index}
        np.savez(folder+'/data'+algorithm+'_'+setup_nm+'_'+setup+'.npz',
                 **data)
    # LOAD AND (POST)PROCESS DATA
    print('Loading data from: ', folder+'/data'+algorithm+'_'+setup_nm +
          '_'+setup+'.npz')
    tmp = np.load(folder+'/data'+algorithm+'_'+setup_nm+'_'+setup+'.npz',
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

    # plot results
    ax1 = ax_final[0]
    ax2 = ax_final[1]
    ax3 = ax_final[2]
    # final figures
    # prop of instances reaching phase 4
    ax_props['ylabel'] = 'Proportion of instances reaching phase 4'
    plt_perf_indicators(values=data['reached_ph'], index_val=val_index,
                        ax=ax1[0], f_props=f_final_prop,
                        ax_props=ax_props, discard=['full', '4'],
                        errorbars=False, plot_individual_values=False)
    # trials to reach phase 4
    ax_props['ylabel'] = 'Number of trials to reach phase 4'
    plt_perf_indicators(values=data['tr_to_ph'],
                        f_props=f_final_prop, ax_props=ax_props,
                        index_val=val_index, ax=ax1[1],
                        reached=data['reached_ph'], discard=['full', '4'],
                        plot_individual_values=plt_ind_vals)
    handles, labels = ax1[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1[0].legend(by_label.values(), by_label.keys())
    ax1[1].set_yscale('log')

    # steps to reach phase 4
    ax_props['ylabel'] = 'Number of steps to reach phase 4 (x1000)'
    plt_perf_indicators(values=data['stps_to_ph'],
                        f_props=f_final_prop, ax_props=ax_props,
                        index_val=val_index, ax=ax2[0],
                        reached=data['reached_ph'], discard=['full', '4'],
                        plot_individual_values=plt_ind_vals)
    # steps to reach final perf
    ax_props['ylabel'] = 'Number of steps to reach final performance (x1000)'
    plt_perf_indicators(values=data['stps_to_perf'],
                        reached=data['reached_perf'],
                        index_val=val_index, ax=ax2[1],
                        f_props=f_final_prop, ax_props=ax_props,
                        plot_individual_values=plt_ind_vals)
    handles, labels = ax2[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2[0].legend(by_label.values(), by_label.keys())

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


def batch_results(algs, setup_vals, markers, tag, setup_nm, folder,
                  limit_tr=True, rerun=False):
    """Runs plot_results function for each of the used variables.
    """
    # Create figures for each of the used algorithms
    for alg in algs:
        print(alg)
        print('xxxxxxxxxxxxxxxxxxxxxx')
        f1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        f2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        f3, ax3 = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        ax = [ax1, ax2, ax3]
        # plot results obtained when training with different second level
        # values (different window/n_ch)
        for ind_setup, setup in enumerate(setup_vals):
            plot_results(folder, alg, setup=setup, setup_nm=setup_nm,
                         limit_ax=False, ax_final=ax, tag=tag,
                         limit_tr=limit_tr,
                         f_final_prop={'color': CLRS[ind_setup],
                                       'label': setup,
                                       'marker': markers[ind_setup]},
                         rerun=rerun)
            if ind_setup == 0:
                f1.savefig(folder + '/final_results_phase_' +
                           alg+'_'+str(limit_tr)+setup+'.svg', dpi=200)
                f2.savefig(folder + '/final_results_steps_' +
                           alg+'_'+str(limit_tr)+setup+'.svg', dpi=200)
                f3.savefig(folder + '/final_results_performance_' +
                           alg+'_'+str(limit_tr)+setup+'.svg', dpi=200)

        f1.savefig(folder + '/final_results_phase_' +
                   alg+'_'+str(limit_tr)+'.svg', dpi=200)
        f2.savefig(folder + '/final_results_steps_' +
                   alg+'_'+str(limit_tr)+'.svg', dpi=200)
        f3.savefig(folder + '/final_results_performance_' +
                   alg+'_'+str(limit_tr)+'.svg', dpi=200)
        plt.close(f1)
        plt.close(f2)
        plt.close(f3)


if __name__ == '__main__':
    plt.close('all')
    # sv_f = '/home/molano/shaping/results_280421/no_shaping/'
    # sv_f = '/home/manuel/shaping/results_280421/'
    sv_f = '/Users/leyreazcarate/Desktop/TFG/shaping/results_280421/'
    RERUN = False
    LEARN = True
    NUM_STEPS = 200000  # 1e5*np.arange(10, 21, 2)
    TH = 0.75
    NUM_RAND = 100000

    plot_separate_figures = True
    plot_all_figs = True
    num_instances = 3
    mean_perf = []
    stages = np.arange(5)
    perf_w = 100
    stg_w = 1000
    conv_w = 50
    rand_act_prob = 0.01
    punish_3_vector = np.linspace(-0.5, 0, 3)
    punish_6_vector = np.linspace(-0.5, 0, 3)
    timing = {'fixation': ('constant', 0),
              'stimulus': ('constant', 300),
              'delay': (0, 100, 300),
              'decision': ('constant', 200)}
    rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.1}
    env_kwargs = {'timing': timing, 'rewards': rewards}
    learning(num_instances, punish_3_vector, sv_f, stages, perf_w, stg_w,
             env_kwargs)
    algs = ['A2C']
    n_ch = ['2', '10', '20']
    markers = ['+', 'x', '1']
    setup_nm = 'n_ch'
    tag = 'stages'
    folder = sv_f+'/large_actObs_space/'
    batch_results(algs=algs, setup_vals=n_ch, markers=markers, tag=tag,
                  setup_nm=setup_nm, folder=folder, limit_tr=False,
                  rerun=RERUN)
    # if plot_separate_figures:
    #     plot_inst_punishment(num_instances, punish_3_vector, conv_w)
    # if plot_all_figs:
    #     plot_figs(punish_6_vector, num_instances, conv_w)
    # print('separate code into functions')
