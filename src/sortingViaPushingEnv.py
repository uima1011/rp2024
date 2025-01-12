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
colours = ['green'] # ['green', 'red']
objectFolders = ['cubes'] # ['signs', 'cubes']
parts = ['cube'] # ['plus', 'cube']

ROBOT_STATE_COUNT = 2 # x and y pos
MAX_OBJECT_COUNT = 2*len(colours)*len(parts) # max 4 of each object type and colour
GOAL_COUNT = len(colours) # red and green
OBJECT_STATE_COUNT = 3 # x, y and rotation arround z pos
GOAL_STATE_COUNT = 3 # x, y and rotation arround z

class sortingViaPushingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	
	def __init__(self):
		super(sortingViaPushingEnv, self).__init__()
		
		self.hdlEnv = HandleEnvironment(RENDER, ASSETS_PATH)
		self.calcReward = CalcReward(self.hdlEnv)
		self.stepCount = 0
		self.episodeCount = 0
		self.counter = 1 # for giving increassing step sizes before timeout
		self.action_space = gym.spaces.Discrete(4) # 4 directions (forward, backward, left, right)
		state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * MAX_OBJECT_COUNT + GOAL_STATE_COUNT * GOAL_COUNT # robot + max objects + goal states
		lowBounds, highBounds = self.hdlEnv.getBoundaries()
		self.observation_space = gym.spaces.Box(low=lowBounds, high=highBounds,
											shape=(state_dim,), dtype=np.float64)

	def step(self, action):
		self.hdlEnv.performAction(action)

		self.reward = self.calcReward.calcReward3()

		self.terminated = self.calcReward.taskFinished()

		self.misbehaviour = self.hdlEnv.checkMisbehaviour() # Task ended unsuccessfully (object falls from table or robot away from table)
		self.timeout, self.counter = self.calcReward.taskTimeout(self.stepCount, self.episodeCount, self.counter) # timeout because robot not fullfiling task
		self.truncated = self.misbehaviour or self.timeout

		observation = self.hdlEnv.getStates()
		info = {'Episode': self.episodeCount, 'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Terminated': self.terminated, 'Truncated': self.truncated, 'Observation': observation}
		pprint(info)
		self.stepCount += 1
		if self.misbehaviour:
			self.reward = -100
		self.reward -= 0.01 # small penalty for faster learning
		return observation, self.reward, self.terminated, self.truncated, info
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.episodeCount += 1 # manualCount
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
		
		info = {'Episode': self.episodeCount, 'Step': self.stepCount, 'Reward': self.reward, 'Action': -1, 'Terminated': self.terminated, 'Truncated': self.truncated, 'Oberservation': observation}
		print("Environment resetted")
		return observation, info
	
	def close(self):
		self.hdlEnv.close()
