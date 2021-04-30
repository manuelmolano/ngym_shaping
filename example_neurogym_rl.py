# -*- coding: utf-8 -*-


import numpy as np
import gym
import matplotlib.pyplot as plt
import ngym_shaping as ng_sh
import warnings
# from stable_baselines.common.policies import LstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C  # ACER, PPO2
warnings.filterwarnings('default')
learn = False
punishment_variation = False
probes = True
# sv_f = '/home/molano/shaping/results_280421/'
# sv_f = '/home/manuel/shaping/results_280421/'
sv_f = '/Users/leyreazcarate/Desktop/TFG/shaping/'
num_steps = 200000  # 1e5*np.arange(10, 21, 2)
num_instances = 3
mean_perf = []
stages = np.arange(5)
th = 0.75
perf_w = 100
stg_w = 1000
timing = {'fixation': ('constant', 0),
          'stimulus': ('constant', 300),
          'delay': (0, 100, 300),
          'decision': ('constant', 200)}
rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.1}
env_kwargs = {'timing': timing, 'rewards': rewards}
for ind in range(num_instances):
    sv_f_inst = sv_f+'/instance_'+str(ind)+'/'
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
        for ind in range(100000):
            action = env.gt_now  # correct action (gt means ground-truth)
            env.step(action)
    env.close()
    # env = ng_sh.envs.DR_stage.shaping(stages=[4], th=th, perf_w=perf_w,
    #                                   stg_w=stg_w, **env_kwargs)
    # env = DummyVecEnv([lambda: env])
    # ng_sh.utils.plotting.plot_env(env, num_trials=100, model=model)
    # data = ng_sh.utils.plotting.run_env(env, num_trials=1000, model=model)
    # perf = np.array(data['perf'])
    # mean_perf.append(np.mean(perf[perf != -1.]))

    # print(mean_perf)
    
### Plot real performance, mean performance and stages
    conv_w = 50
    plt.close('all')
    f, ax = plt.subplots(nrows=2, sharex=True)
    ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                                  fkwargs={'c': 'tab:red'},
                                                  legend=False, zline=False,
                                                  metric_name='performance',
                                                  window=conv_w)
    ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                                  fkwargs={'c': 'tab:blue'},
                                                  legend=False, zline=False,
                                                  metric_name='real_performance',
                                                  window=conv_w)
    ax[0].axhline(y=th, linestyle='--', color='k')
    ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                                  fkwargs={'c': 'tab:blue'},
                                                  legend=False, zline=False,
                                                  metric_name='stage',
                                                  window=conv_w)
    f.savefig(sv_f_inst+'training.png', dpi=300)

### Plot real performance, mean performance and stages with different punishments
# plot of each punishment and each instance
if punishment_variation:
    plt.close('all')
    for ind in range(num_instances):
        sv_f_inst = sv_f+'/instance_'+str(ind)+'/'
        for i in np.arange(-0.5, 0, 0.1):
            i = round(i, 2)
            rewards = {'abort': -0.1, 'correct': +1., 'fail': i}
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
                for ind in range(100000):
                    action = env.gt_now  # correct action (gt means ground-truth)
                    env.step(action)
            env.close()
            conv_w = 50
            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].set_title('Punishment' + str(i))
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                                          fkwargs={'c': 'tab:red'},
                                                          legend=False, zline=False,
                                                          metric_name='performance',
                                                          window=conv_w)
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0],
                                                          fkwargs={'c': 'tab:blue'},
                                                          legend=False, zline=False,
                                                          metric_name='real_performance',
                                                          window=conv_w)
            ax[0].axhline(y=th, linestyle='--', color='k')
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[1],
                                                          fkwargs={'c': 'tab:blue'},
                                                          legend=False, zline=False,
                                                          metric_name='stage',
                                                          window=conv_w)
            f.savefig(sv_f_inst+'training with punishment' + str(i) +'.png', dpi=300)
                
        
if probes:
    plt.close('all')
    for ind in range(num_instances):
        sv_f_inst = sv_f+'/instance_'+str(ind)+'/'
        for index, punishment in enumerate(np.arange(-0.5, 0, 0.1)):
            punishment = round(punishment, 2)
            rewards = {'abort': -0.1, 'correct': +1., 'fail': punishment}
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
                for ind in range(100000):
                    action = env.gt_now  # correct action (gt means ground-truth)
                    env.step(action)
            env.close()
            conv_w = 50
            f, ax = plt.subplots(nrows=2, ncols=5, sharex=True)
            cols = ['Punishment {}'.format(round(col,2)) for col in np.arange(-0.5, 0, 0.1)]
            rows = ['Performance', 'Stages']
            for axes, col in zip(ax[0], cols):
                axes.set_title(col)
            for axes, row in zip(ax[:,0], rows):
                axes.set_ylabel(row, rotation=0, size='large')
            f.tight_layout()
            # ax.set_title('Punishment' + str(punishment))
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0:index],
                                                          fkwargs={'c': 'tab:red'},
                                                          legend=False, zline=False,
                                                          metric_name='performance',
                                                          window=conv_w)
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[0:index],
                                                          fkwargs={'c': 'tab:blue'},
                                                          legend=False, zline=False,
                                                          metric_name='real_performance',
                                                          window=conv_w)
            ax[0].axhline(y=th, linestyle='--', color='k')
            ng_sh.utils.plotting.plot_rew_across_training(folder=sv_f_inst, ax=ax[1:index],
                                                          fkwargs={'c': 'tab:blue'},
                                                          legend=False, zline=False,
                                                          metric_name='stage',
                                                          window=conv_w)
            f.savefig(sv_f_inst+'training with punishment' + str(punishment) +'.png', dpi=300)
                
    



