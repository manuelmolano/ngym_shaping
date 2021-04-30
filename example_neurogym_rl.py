# -*- coding: utf-8 -*-


import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import ngym_shaping as ng_sh
from ngym_shaping.utils import plotting as plot
import warnings
# from stable_baselines.common.policies import LstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C  # ACER, PPO2
warnings.filterwarnings('default')
rerun = True
learn = False
punishment_variation = True
probes = False
plot_figs = True
sv_f = '/home/molano/shaping/results_280421/'
# sv_f = '/home/manuel/shaping/results_280421/'
# sv_f = '/Users/leyreazcarate/Desktop/TFG/shaping/'
num_steps = 200000  # 1e5*np.arange(10, 21, 2)
num_instances = 3
mean_perf = []
stages = np.arange(5)
th = 0.75
perf_w = 100
stg_w = 1000
conv_w = 50
rand_act_prob = 0.01
timing = {'fixation': ('constant', 0),
          'stimulus': ('constant', 300),
          'delay': (0, 100, 300),
          'decision': ('constant', 200)}
rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.1}
env_kwargs = {'timing': timing, 'rewards': rewards}

### Plot real performance, mean performance and stages with different punishments
# plot of each punishment and each instance
plt.close('all')
for i_i in range(num_instances):
    for pun in np.linspace(-0.5, 0, 3):  # TODO: define pun mat
        sv_f_inst = sv_f+'/pun_'+str(round(pun, 2))+'_inst_'+str(i_i)+'/'
        print(sv_f_inst)
        if not os.path.exists(sv_f_inst+'/bhvr_data_all.npz') or rerun:
            rewards = {'abort': -0.1, 'correct': +1., 'fail': pun}
            env_kwargs = {'timing': timing, 'rewards': rewards}
            env = ng_sh.envs.DR_stage.shaping(stages=stages, th=th, perf_w=perf_w,
                                              stg_w=stg_w, sv_folder=sv_f_inst,
                                              sv_per=stg_w, **env_kwargs)
            if learn:
                env = DummyVecEnv([lambda: env])
                # Define model
                model = A2C(LstmPolicy, env, verbose=1,
                            policy_kwargs={'feature_extraction': "mlp"})
                # Train model
                model.learn(total_timesteps=num_steps, log_interval=10e10)
            else:
                env.reset()
                for ind in range(100000):  # TODO: define
                    if np.random.rand() < (rand_act_prob-pun):
                        action = np.random.randint(0, 3)
                    else:
                        action = env.gt_now  # correct action (gt = ground-truth)
                    env.step(action)
            env.close()
            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].set_title('Punishment' + str(round(pun, 2)))
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                          fkwargs={'c': 'tab:red'},
                                          legend=False, zline=False,
                                          metric_name='performance',
                                          window=conv_w)
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                          fkwargs={'c': 'tab:blue'},
                                          legend=False, zline=False,
                                          metric_name='real_performance',
                                          window=conv_w)
            ax[0].axhline(y=th, linestyle='--', color='k')
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                          fkwargs={'c': 'tab:blue'},
                                          legend=False, zline=False,
                                          metric_name='stage',
                                          window=conv_w)
            f.savefig(sv_f_inst+'pun_'+str(round(pun, 2))+'.png', dpi=300)
if plot_figs:
    plt.close('all')
    f, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fkwargs = {'c': 'tab:blue'}
    for i_p, pun in enumerate(np.linspace(-0.5, 0, 6)):  # TODO: define pun mat
        for i_i in range(num_instances):
            pun_str = str(round(pun, 2))
            sv_f_inst = sv_f+'/pun_'+pun_str+'_inst_'+str(i_i)+'/'
            fkwargs['alpha'] = 1-1/(i_p+2)
            fkwargs['label'] = 'pun = '+pun_str if i_i == 0 else ''
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                          fkwargs=fkwargs, legend=False,
                                          zline=False, window=conv_w,
                                          metric_name='real_performance')
            ax[0].axhline(y=th, linestyle='--', color='k')
            plot.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                          fkwargs=fkwargs, legend=False,
                                          zline=False, window=conv_w,
                                          metric_name='stage')
    f.tight_layout()
    ax[0].legend()
    f.savefig(sv_f+'/pun_'+pun_str+'_all_insts.png', dpi=300)


if __name__ == '__main__':
    print('separate code into functions')  # TODO: do
