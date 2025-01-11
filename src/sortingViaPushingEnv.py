import gymnasium as gym
import numpy as np

from handleEnvironment import HandleEnvironment, CalcReward
from pprint import pprint
# Setup Simulation
RENDER = True
ASSETS_PATH = "/home/group1/workspace/assets"

# Train:
MAX_STEPS = 1000

# Environment
colours = ['green', 'red']
objectFolders = ['signs', 'cubes']
parts = ['plus', 'cube']

ROBOT_STATE_COUNT = 2 # x and y pos
MAX_OBJECT_COUNT = 3*len(colours)*len(parts) # max 4 of each object type and colour
GOAL_COUNT = len(colours) # red and green
OBJECT_STATE_COUNT = 3 # x, y and rotation arround z pos
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
		self.reward = self.calcReward.calcReward()
		self.terminated = self.hdlEnv.checkMisbehaviour() # TODO 
		if self.stepCount >= MAX_STEPS-1:
			self.truncated = True
		else:
			self.truncated = False # TODO ist das nicht sowieso schon false?
		
		

		observation = self.hdlEnv.getStates()
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Terminated': self.terminated, 'Truncated': self.truncated, 'Observation': observation}
		pprint(info)
		self.stepCount += 1
		return observation, self.reward, self.terminated, self.truncated, info
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.stepCount = 0
		self.terminated = False
		self.truncated  = False
		self.reward = 0
		self.hdlEnv.resetEnvironment()
		self.hdlEnv.robotToStartPose()
		self.hdlEnv.spawnGoals()
		self.hdlEnv.spawnObjects()
		self.calcReward.reset()
		
		observation = self.hdlEnv.getStates() # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
		
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': -1, 'Terminated': self.terminated, 'Truncated': self.truncated, 'Oberservation': observation}
		print("Environment resetted")
		return observation, info
