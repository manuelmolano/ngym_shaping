#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:44:59 2021

@author: molano
"""
import numpy as np

import ngym_shaping as ngym
from ngym_shaping import spaces
from ngym_shaping.wrappers.block import ScheduleEnvs_condition as sch_cond
from ngym_shaping.utils.scheduler import SequentialSchedule_condition as sq_sch_cnd
from ngym_shaping.wrappers import mean_perf
from ngym_shaping.wrappers import monitor


class Evolve_grid(ngym.TrialEnv):
    """Grid world.

    Agents have to find food.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
    """
    metadata = {
        'paper_link': '',
        'paper_name': '',
        'tags': ['perceptual']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, stage=4,
                 max_num_rep=3):
        super().__init__(dt=dt)
        self.abort = False
        self.choices = [1, 2, 3 ,4]  # right or left options
        # cohs specifies the amount of evidence (modulated by stim_scale)
        # cohs = How much different are right and left stimulus
        self.sigma = 0  # How much noise is applied
        # until stage 4 there is not noise
        self.stage = stage
        self.first_counts = False  # first answer is not penalized
        # this only happens in stage 1
        self.max_num_rep = max_num_rep
        self.rep_counter = 0
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}

        if stage == 0:
            self.rewards = {'abort': -0.1, 'correct': +1., 'fail': +1.}
            # differences between the 2 stimuli
            self.cohs = np.array([0])*stim_scale
        elif stage == 1:
            self.first_counts = True
        elif stage == 3:
            delays = (0, 300, 1000)  # delays are introduced
        elif stage == 4:
            self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stim_scale
            delays = (0, 300, 1000)
            self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        if rewards:
            self.rewards.update(rewards)
        self.timing = {'fixation': 0, 'stimulus': 300, 'delay': delays,
                       'decision': 400}
        if timing:
            self.timing.update(timing)

        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.real_performance = False

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma': self.sigma,
        }
        trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)

        # define observations
        self.set_ob([1, 0, 0], 'fixation')
        stim = self.view_ob('stimulus')
        stim[:, 0] = 1
        stim[:, 1:] = (1 - trial['coh']/100)/2
        stim[:, trial['ground_truth']] = (1 + trial['coh']/100)/2
        stim[:, 1:] +=\
            self.rng.randn(stim.shape[0], 2) * trial['sigma']

        self.set_ob([1, 0, 0], 'delay')

        self.set_groundtruth(trial['ground_truth'], 'decision')
        if self.stage == 1:
            self.first_action_flag = True
        self.real_performance = False
        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision') and action != 0:
            # allow change of mind
            if self.stage == 0:
                self.count(action)
                # reward is 0 if it repeating more than it should
                reward = 0 if abs(self.rep_counter) > self.max_num_rep\
                    else self.rewards['correct']
                self.performance = reward == self.rewards['correct']
            elif action == gt:
                # correct behaviour: network action = gound truth
                reward = self.rewards['correct']
                self.performance = 1
            elif action == 3 - gt and self.stage != 1:
                # 3-action is the other act
                # not correct behaviour
                reward = self.rewards['fail']
            if self.stage != 1:
                new_trial = True
                self.real_performance = self.performance
            else:
                new_trial = action == gt
                if self.first_action_flag:
                    self.real_performance = action == gt
                    self.first_action_flag = False
        info = {'new_trial': new_trial, 'gt': gt,
                'real_performance': self.real_performance, 'stage': self.stage}
        return self.ob_now, reward, False, info

    def count(self, action):
        '''
        counts number of repetitions of action.
        '''
        # action_1_minus_1 is juss just the original action expressed as -1 and 1
        # instead of 1 and 2
        action_1_minus_1 = action - 2/action
        if np.sign(self.rep_counter) == np.sign(action_1_minus_1):
            self.rep_counter += action_1_minus_1  # add to counter
        else:
            self.rep_counter = action_1_minus_1   # reset counter


def shaping(stages=None, th=0.75, perf_w=20, stg_w=100, sv_folder=None,
            sv_per=10e5, **env_kwargs):
    """
    Put environments together.

    stages : list, optional
        list of stages to add to the shaping. If None stages=np.arange(4) (None)
    th : float, optional
        performance threshold to increase the stage (0.75)
    perf_w : int, optional
        window to compute performance (20)
    stg_w: int, optional
        minimum number of trials in each stage (100)

    """
    def cond(action, obs, rew, info):
        return info['mean_perf'] > th
    if stages is None:
        stages = np.arange(5)
    envs = []
    for stg in stages:
        env = DR_stage(stage=stg, **env_kwargs)
        env = mean_perf.MeanPerf(env, perf_w=perf_w)
        if sv_folder is not None:
            env = monitor.Monitor(env, folder=sv_folder, sv_fig=False, fig_type='',
                                  name='stg_'+str(float(stg)), sv_per=sv_per)
        envs.append(env)
    schedule = sq_sch_cnd(n=len(envs), cond=cond, w=stg_w)
    # schedule decides when to change stage:
    # if the condition (mean_perf > threshold) is met and last stage (4) has not
    # been reached and a certain number of trials have been elapsed in this
    # stage, then stage is increased.
    env = sch_cond(envs, schedule, env_input=False)
    return env


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
    timing = {'decision': 1000, 'stimulus': 100}
    rewards = {'abort': -0.1, 'correct': +1., 'fail': -1.}
    env_kwargs = {'timing': timing, 'rewards': rewards}
    # env = Shaping(stage=1, timing=timing, rewards=rewards)
    for stg in range(4):
        env = shaping(stages=[stg], th=0.75, perf_w=20, stg_w=100, **env_kwargs)
        env.reset()
        # real_perf = []
        # perf = []
        # rew = []
        # act = []
        # gt = []
        # stg = []
        # for ind in range(100):
        #     action = env.gt_now  # correct action (gt means ground-truth)
        #     ob_now, reward, _, info = env.step(action)
        #     real_perf.append(info['real_performance'])
        #     rew.append(reward)
        #     act.append(action)
        #     gt.append(info['gt'])
        #     stg.append(env.stage)
        #     if info['new_trial']:
        #         perf.append(info['performance'])
        #     else:
        #         perf.append(0)
        f = ngym.utils.plot_env(env, fig_kwargs={'figsize': (6, 6)}, num_steps=50,
                                ob_traces=['Fixation cue', 'Stim 1', 'Stim 2'])
        f.savefig('/home/manuel/Escritorio/shaping_stg_'+str(stg)+'.png')
        f.savefig('/home/manuel/Escritorio/shaping_stg_'+str(stg)+'.svg')
    # f, ax = plt.subplots(nrows=5, ncols=1, sharex=True)
    # ax[0].plot(rew, label='reward')
    # ax[0].set_title('reward')
    # ax[1].plot(np.array(perf), label='perf')
    # ax[1].set_title('perf')
    # ax[2].plot(np.array(real_perf), label='real perf')
    # ax[2].set_title('real perf')
    # ax[3].plot(np.array(act), label='actions')
    # ax[3].plot(np.array(gt), '--', label='gt')
    # ax[3].set_title('actions')
    # ax[3].legend()
    # ax[4].plot(np.array(stg), label='stages')
    # ax[4].set_title('stages')
    # plt.legend()
