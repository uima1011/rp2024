'''
Skript to train a model for the sorting via pushing task.
'''

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

# initialize environment

env = svpEnv()

model = DQN(cfg['policy'],            								# policy network
    env,                            								# environment
    gamma = cfg['DQN']['gamma'],                     				# discount faktor
    learning_rate = float(cfg['DQN']['learning_rate']),             # stable learning rate
    buffer_size = cfg['DQN']['buffer_size'],             			# replay buffer size
    batch_size = cfg['DQN']['batch_size'],                  		# standard for DQN
    train_freq = cfg['DQN']['train_freq'],                   		# train every x steps
    target_update_interval = cfg['DQN']['target_update_interval'],	# update target network every x steps
    exploration_fraction = cfg['DQN']['exploration_fraction'],      # exploration fraction
    exploration_final_eps = cfg['DQN']['exploration_final_eps'],    # minimum epsilon
    verbose = cfg['DQN']['verbose'],                      			# extensive output
    tensorboard_log = logDir          								# tensorboard log directory
)

# training

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=cfg['resetTimesteps'], tb_log_name=MODEL)
    model.save(f"{modelsDir}/{TIMESTEPS * iters}")

