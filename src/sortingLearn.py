from stable_baselines3 import PPO
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
import time


TIMESTEPS = 10000
MODEL = "PPO"

modelsDir = f"/home/group1/workspace/data/models/{MODEL}"
if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
logDir = "/home/group1/workspace/data/logs"
if not os.path.exists(logDir):
    os.makedirs(logDir)

env = svpEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logDir)


iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
	model.save(f"{modelsDir}/{TIMESTEPS*iters}")