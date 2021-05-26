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
def learning(sv_f, pun_vector, stages, perf_w=100, stg_w=1000, rollout=5,
             gamma=0.5, num_instances=3, env_kwargs={}):
    for i_i in range(num_instances):
        for pun in pun_vector:
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
                                **{'n_steps': rollout, 'gamma': gamma})
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


if __name__ == '__main__':
    plt.close('all')
    rollout = 5
    gamma = 0.5
    # sv_f = '/home/molano/shaping/results_280421/no_shaping/'
    # sv_f = '/home/molano/shaping/results_280421/shaping_long_tr_one_agent/'
    # sv_f = '/home/molano/shaping/results_280421/' +\
    #     'no_shaping_long_tr_one_agent_stg_4_nsteps_'+str(rollout)+'/'
    # sv_f = '/home/molano/shaping/results_280421/shaping_diff_punishment/'
    sv_f = '/home/molano/shaping/results_280421/' +\
        'shaping_'+str(rollout)+'_'+str(gamma)+'/'

    RERUN = True
    LEARN = True
    NUM_STEPS = 300000  # 1e5*np.arange(10, 21, 2)
    TH = 0.75
    NUM_RAND = 100000

    num_instances = 20
    mean_perf = []
    stages = np.arange(2)  # np.array([4])  # np.arange(5)
    perf_w = 100
    stg_w = 1000
    conv_w = 50
    rand_act_prob = 0.01
    pun_vector = [0]  # np.linspace(-1.0, 0., 5)  # np.linspace(-0.5, 0, 3)
    timing = {'fixation': ('constant', 200),
              'stimulus': ('constant', 400),
              'delay': (0, 1000, 3000),
              'decision': ('constant', 200)}
    env_kwargs = {'timing': timing}
    learning(sv_f=sv_f, num_instances=num_instances, pun_vector=pun_vector,
             stages=stages, perf_w=perf_w, stg_w=stg_w, rollout=rollout,
             gamma=gamma, env_kwargs=env_kwargs)
    # TODO: plot performances using analysis_rl.py fns
