# -*- coding: utf-8 -*-


import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import ngym_shaping as ng_sh
from ngym_shaping.utils import plotting as plot
import warnings
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C  # ACER, PPO2
warnings.filterwarnings('default')


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
                    model = A2C(LstmPolicy, env, verbose=1, n_cpu_tf_sess=1,
                                policy_kwargs={'feature_extraction': "mlp"},
                                **{'n_steps': 10})
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


if __name__ == '__main__':
    plt.close('all')
    # sv_f = '/home/molano/shaping/results_280421/no_shaping/'
    # sv_f = '/home/molano/shaping/results_280421/shaping_long_tr_one_agent/'
    sv_f = '/home/molano/shaping/results_280421/' +\
        'no_shaping_long_tr_one_agent_stg_4_nsteps_10/'
    # sv_f = '/home/molano/shaping/results_280421/shaping_diff_punishment/'
    RERUN = True
    LEARN = True
    NUM_STEPS = 300000  # 1e5*np.arange(10, 21, 2)
    TH = 0.75
    NUM_RAND = 100000

    plot_separate_figures = True
    plot_all_figs = True
    num_instances = 3
    mean_perf = []
    stages = [4]  # np.arange(4)  #np.array([4])  # np.arange(5)
    perf_w = 100
    stg_w = 1000
    conv_w = 50
    rand_act_prob = 0.01
    punish_3_vector = np.linspace(-1.0, 0., 5)  # np.linspace(-0.5, 0, 3)
    timing = {'fixation': ('constant', 200),
              'stimulus': ('constant', 400),
              'delay': (0, 1000, 3000),
              'decision': ('constant', 200)}
    rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.1}
    env_kwargs = {'timing': timing, 'rewards': rewards}
    learning(num_instances, punish_3_vector, sv_f, stages, perf_w, stg_w,
             env_kwargs)
    if plot_separate_figures:
        plot_inst_punishment(num_instances, punish_3_vector, conv_w)
    if plot_all_figs:
        plot_figs(punish_3_vector, num_instances, conv_w)
    print('separate code into functions')
