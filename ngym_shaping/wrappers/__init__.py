from ngym_shaping.wrappers.monitor import Monitor
from ngym_shaping.wrappers.noise import Noise
from ngym_shaping.wrappers.pass_reward import PassReward
from ngym_shaping.wrappers.pass_action import PassAction
from ngym_shaping.wrappers.reaction_time import ReactionTime
from ngym_shaping.wrappers.side_bias import SideBias
from ngym_shaping.wrappers.block import RandomGroundTruth
from ngym_shaping.wrappers.block import ScheduleAttr
from ngym_shaping.wrappers.block import ScheduleEnvs
from ngym_shaping.wrappers.block import TrialHistoryV2

ALL_WRAPPERS = {'Monitor-v0': 'ngym_shaping.wrappers.monitor:Monitor',
                'Noise-v0': 'ngym_shaping.wrappers.noise:Noise',
                'PassReward-v0': 'ngym_shaping.wrappers.pass_reward:PassReward',
                'PassAction-v0': 'ngym_shaping.wrappers.pass_action:PassAction',
                'ReactionTime-v0':
                    'ngym_shaping.wrappers.reaction_time:ReactionTime',
                'SideBias-v0': 'ngym_shaping.wrappers.side_bias:SideBias',
                }

def all_wrappers():
    return sorted(list(ALL_WRAPPERS.keys()))
