#from stable_baselines3 import PPO
#import os
#from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
#
#TIMESTEPS = 10000
#MODEL = "PPO_Hyperparameter_neu"
#
#modelsDir = f"/home/group1/workspace/data/models/{MODEL}"
#if not os.path.exists(modelsDir):
#    os.makedirs(modelsDir)
#logDir = f"/home/group1/workspace/data/logs/{MODEL}"
#if not os.path.exists(logDir):
#    os.makedirs(logDir)
#
#env = svpEnv()
#
##model = PPO('MlpPolicy', env, gamma = 0.99, ent_coef=0.01, verbose=1, tensorboard_log=logDir)
##model = PPO.load(f"/home/group1/workspace/data/models/{MODEL}/100000.zip", env, gamma = 0.99, ent_coef=0.01, verbose=1, tensorboard_log=logDir) # use existing model
#model = PPO(
#    'MlpPolicy',
#    env,
#    gamma=0.95,  # Fokus auf kurzfristigere Belohnungen
#    ent_coef=0.02,  # Erhöhte Exploration
#    learning_rate=1e-4,  # Stabileres Lernen
#    n_steps=4096,  # Größere Batchgröße für stabilere Gradienten
#    clip_range=0.1,  # Engere Clipping-Range für stabilere Updates
#    max_grad_norm=1.0,  # Erhöhte Toleranz für Gradientenänderungen
#    batch_size=128,  # Größere Batchgröße für optimierte Lernzyklen
#    n_epochs=15,  # Mehr Trainingsepochen pro Update
#    verbose=1,
#    tensorboard_log=logDir
#)
#
#iters = 0
#while True:
#	iters += 1
#	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
#	model.save(f"{modelsDir}/{TIMESTEPS*iters}")

# python3 -m tensorboard.main --logdir=data/logs
     

from stable_baselines3 import DQN
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv

TIMESTEPS = 10000
MODEL = "DQN_normiert_push_after_to_object_4"

modelsDir = f"/home/group1/workspace/data/models/{MODEL}"
if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
logDir = "/home/group1/workspace/data/logs"
if not os.path.exists(logDir):
    os.makedirs(logDir)

# Umgebung initialisieren
env = svpEnv()

model = DQN.load("../data/models/DQN_normiert_push_after_to_object_3/70000.zip",
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
# Model laden
#model = DQN.load(f"/home/group1/workspace/data/models/DQN/70000.zip",
#    env, 
#    gamma=0.99, 
#    learning_rate=1e-4,  # Stabileres Lernen
#    buffer_size=100000,  # Replay Buffer Größe
#    batch_size=64,  # Standardwert für DQN
#    train_freq=4,  # Training nach jeder 4. Aktion
#    target_update_interval=1000,  # Zielnetzwerk-Update-Intervall
#    exploration_fraction=0.1,  # 10% der Trainingszeit für Exploration
#    exploration_final_eps=0.02,  # Minimaler Explorationswert
#    verbose=1, 
#    tensorboard_log=logDir
#)

# Training
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
    model.save(f"{modelsDir}/{TIMESTEPS * iters}")

