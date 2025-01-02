import gymnasium as gym

import numpy as np
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine

from handleEnvironment import HandleEnvironment, CalcReward

# Setup Simulation
RENDER = True
ASSETS_PATH = "/home/group1/workspace/assets"

# Train:
MAX_STEPS = 100

# Environment
colours = ['green', 'red']
objectFolders = ['signs', 'cubes']
parts = ['plus', 'cube']

ROBOT_STATE_COUNT = 2 # x and y
MAX_OBJECT_COUNT = 4*len(colours)*len(parts) # max 4 of each object type and colour
GOAL_COUNT = len(colours) # red and green
OBJECT_STATE_COUNT = 3 # x, y and rotation arround z
GOAL_STATE_COUNT = 3 # x, y and rotation arround z

class sortingViaPushingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	
	def __init__(self):
		super(sortingViaPushingEnv, self).__init__()
		self.action_space = gym.spaces.Discrete(4) # 4 directions (forward, backward, left, right)
		state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * MAX_OBJECT_COUNT + GOAL_STATE_COUNT * GOAL_COUNT # robot + max objects + goal states
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(state_dim,), dtype=np.float64)
		self.hdlEnv = HandleEnvironment(RENDER, ASSETS_PATH)
		self.calcReward = CalcReward(self.hdlEnv)
		self.stepCount = 0
		
	def step(self, action):
		self.hdlEnv.performAction(action)
        # get deltaReward
		self.totalReward = self.calcReward.calcReward()
		info = {'Step': self.stepCount, 'Reward': {'total': self.totalReward, 'previous': self.prevReward}, 'action': action} # additional debug informations
		self.reward = self.totalReward - self.prevReward
		self.prevReward = self.totalReward
		
		self.done = self.hdlEnv.checkMisbehaviour() # TODO turn True if object falls from table and give ultra bad reward
		if self.stepCount >= MAX_STEPS-1:
			truncated = True
		else:
			truncated = False
		self.stepCount += 1
		observation = self.hdlEnv.getStates()
		return observation, self.reward, self.done, self.truncated, info
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.stepCount = 0
		self.done = False
		self.truncated  = False
		self.prevReward = 0
		self.hdlEnv.resetEnvironment()
		self.hdlEnv.robotToStartPose()
		self.hdlEnv.spawnGoals()
		self.hdlEnv.spawnObjects()
		
        # create observation
		observation = self.hdlEnv.getStates() # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
		info = {}
		return observation, info