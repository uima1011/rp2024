from stable_baselines3 import PPO, DQN
from stable_baselines3.common.policies import ActorCriticPolicy
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
from torch import nn

class CustomMLPPolicy(ActorCriticPolicy):
	def __init__(self, *args, **kwargs):
		super(CustomMLPPolicy, self).__init__(*args, **kwargs,
											  net_arch=[dict(pi=[256, 256], vf=[256, 256])],
											  activation_fn= nn.ReLU,)

TIMESTEPS = 10000
MODEL = "DQN_normedRewards_Symmetric_absPositions"

modelsDir = f"/home/group1/workspace/data/models/{MODEL}"
if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
logDir = "/home/group1/workspace/data/logs"
if not os.path.exists(logDir):
    os.makedirs(logDir)

env = svpEnv()

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logDir, device='cuda')
# model = PPO.load(f"/home/group1/workspace/data/models/{MODEL}/10000.zip", env=env, verbose=1, tensorboard_log=logDir) # use existing model

iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
	model.save(f"{modelsDir}/{TIMESTEPS*iters}")
	