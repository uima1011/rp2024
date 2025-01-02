from stable_baselines3 import PPO
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv

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

model = PPO('MlpPolicy', env, gamma = 0.99, ent_coef=0.01, verbose=1, tensorboard_log=logDir)
# model = PPO.load("/home/group1/workspace/data/models/{MODEL}/XXXXX.zip", env=env, verbose=1, tensorboard_log=logDir) # use existing model

iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
	model.save(f"{modelsDir}/{TIMESTEPS*iters}")