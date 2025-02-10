'''
Skript to train a model for the sorting via pushing task.
'''
# python3 -m tensorboard.main --logdir=data/logs

from stable_baselines3 import DQN
import os
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
import config

cfg = config.load()
cfg = cfg['train']

TIMESTEPS = cfg['timesteps']
MODEL = cfg['name']

modelsDir = os.path.join(cfg['dir'], 'models', MODEL)
logDir = os.path.join(cfg['dir'], "logs", MODEL)

if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)

if not os.path.exists(logDir):
    os.makedirs(logDir)

# Umgebung initialisieren

env = svpEnv()


model = DQN(cfg['policy'],            								# Policy-Netzwerk
    env,                            								# Umgebung
    gamma = cfg['DQN']['gamma'],                     				# Diskontierungsfaktor
    learning_rate = float(cfg['DQN']['learning_rate']),             # Stabileres Lernen
    buffer_size = cfg['DQN']['buffer_size'],             			# Replay Buffer Größe
    batch_size = cfg['DQN']['batch_size'],                  		# Standardwert für DQN
    train_freq = cfg['DQN']['train_freq'],                   		# Training nach jeder 4. Aktion
    target_update_interval = cfg['DQN']['target_update_intervall'],	# Zielnetzwerk-Update-Intervall
    exploration_fraction = cfg['DQN']['exploration_fraction'],      # 10% der Trainingszeit für Exploration
    exploration_final_eps = cfg['DQN']['exploration_final_eps'],    # Minimaler Explorationswert
    verbose = cfg['DQN']['verbose'],                      			# Ausführliche Ausgabe
    tensorboard_log = logDir          								# Tensorboard-Log-Verzeichnis
)

# Training

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=cfg['resetTimesteps'], tb_log_name=MODEL)
    model.save(f"{modelsDir}/{TIMESTEPS * iters}")

