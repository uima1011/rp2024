'''
Skript to train a model for the sorting via pushing task.
'''
# python3 -m tensorboard.main --logdir=data/logs

from stable_baselines3 import DQN
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv

TIMESTEPS = 10000
MODEL = "DQN_normiert_small_observationspace_corrected_padding"

modelsDir = f"/home/group1/workspace/data/models/{MODEL}"
if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
logDir = "/home/group1/workspace/data/logs"
if not os.path.exists(logDir):
    os.makedirs(logDir)

# Umgebung initialisieren

env = svpEnv()

model = DQN('MlpPolicy', 
    env, 
    gamma=0.99, 
    learning_rate=1e-4,  # Stabileres Lernen
    buffer_size=100000,  # Replay Buffer Größe
    batch_size=64,  # Standardwert für DQN
    train_freq=4,  # Training nach jeder 4. Aktion
    target_update_interval=1000,  # Zielnetzwerk-Update-Intervall
    exploration_fraction=0.1,  # 10% der Trainingszeit für Exploration
    exploration_final_eps=0.02,  # Minimaler Explorationswert
    verbose=1, 
    tensorboard_log=logDir
)

# Training

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{modelsDir}/{TIMESTEPS * iters}")

