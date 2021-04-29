# -*- coding: utf-8 -*-


import numpy as np
import gym
import ngym_shaping as ng_sh
import warnings
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C  # ACER, PPO2
warnings.filterwarnings('default')
sv_f = '/home/molano/shaping/results_280421/'
# sv_f = '/home/manuel/CV-learning/results_280421/'
num_steps = [2000000]  # 1e5*np.arange(10, 21, 2)
num_instances = 3
mean_perf = []
stages = np.arange(5)
th = 0.75
perf_w = 100
stg_w = 100
timing = {'fixation': ('constant', 0),
          'stimulus': ('constant', 300),
          'delay': (0, 100, 300),
          'decision': ('constant', 200)}
rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.1}  # no punishment
env_kwargs = {'timing': timing, 'rewards': rewards}
for n_stps in num_steps:
    print('xxxxxxxxxxxxxxxxxx')
    print(n_stps)
    for ind in range(num_instances):
        env = ng_sh.envs.DR_stage.shaping(stages=stages, th=th, perf_w=perf_w,
                                          stg_w=stg_w, sv_folder=sv_f,
                                          **env_kwargs)
        env = DummyVecEnv([lambda: env])
        # Define model
        model = A2C(LstmPolicy, env, verbose=1,
                    policy_kwargs={'feature_extraction': "mlp"})
        # Train model
        model.learn(total_timesteps=int(n_stps), log_interval=10e10)
        data = ng_sh.utils.plotting.run_env(env, num_trials=1000, model=model)
        perf = np.array(data['perf'])
        mean_perf.append(np.mean(perf[perf != -1.]))

    print(mean_perf)
