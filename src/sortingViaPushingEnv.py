'''
	creates gymnasium environment for RL:
	- action_space: 4 directions (forward, backward, left, right)
	- observation_space: robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
	- reward: reward for each step
	- step: perform action and return observation, reward, terminated, truncated, info
	- reset: reset environment and return observation, info

	Additional functions for calculating behaviour of agent:
	- _computeDistances: compute distances between objects and goals
	- score: how far where the objects pushed by the agent
		- logScoreAllObjects: log the score of all objects for the agent
		- logScore: log the score of one object for the agent
'''

import gymnasium as gym
import numpy as np
import math

from handleEnvironment import HandleEnvironment, CalcReward
import config

cfg = config.load()
cfg = cfg['env']

# Setup Simulation
RENDER = cfg['render']
ASSETS_PATH = cfg['assetsPath']

# Train:
MAX_STEPS = cfg['maxSteps']

# Environment
colours = cfg['colours']
objectFolders = cfg['objects']['dirs']
parts = cfg['objects']['parts']

ROBOT_STATE_COUNT = cfg['robot']['states'] # x and y
MAX_OBJECT_COUNT = cfg['objects']['number']*len(colours)*len(parts) # max 4 of each object type and colour
GOAL_COUNT = len(colours) # red and green
OBJECT_STATE_COUNT = cfg['objects']['states'] # x, y and rotation arround z
GOAL_STATE_COUNT = cfg['goals']['states'] # x, y and rotation arround z

class sortingViaPushingEnv(gym.Env):
	"""Custom Environment that follows gym interface"""
	
	def __init__(self):
		super(sortingViaPushingEnv, self).__init__()
		self.action_space = gym.spaces.Discrete(cfg['actions']) # 4 directions (forward, backward, left, right)
		state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * MAX_OBJECT_COUNT + GOAL_STATE_COUNT * GOAL_COUNT # robot + max objects + goal states
		#state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * 1 + GOAL_STATE_COUNT * 1 # robot + max objects + goal states
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(state_dim,), dtype=np.float64)
		self.hdlEnv = HandleEnvironment(RENDER, ASSETS_PATH)
		self.calcReward = CalcReward(self.hdlEnv)
		self.stepCount = 0
		self.startDistance = None
		self.score = 0
		
	def _computeDistances(self, positions):
		def dist(a, b): 
			return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

		# get real goal positions (only non dummy values)
		greenGoalPos = next((v for v in positions['goal_green'].values() if v != [None, None, None]), [0,0,0])
		redGoalPos   = next((v for v in positions['goal_red'].values()   if v != [None, None, None]), [0,0,0])

		# sum up distances
		distances = {}
		for obj_type in ['plus_green', 'cube_green']:
			distances[obj_type] = sum(dist(v, greenGoalPos)
									for v in positions[obj_type].values() if v != [None, None, None])
		for obj_type in ['plus_red', 'cube_red']:
			distances[obj_type] = sum(dist(v, redGoalPos)
									for v in positions[obj_type].values() if v != [None, None, None])
		return distances, sum(distances.values())

	def logScoreAllObjects(self):
		'''Log the score of all objects of the agent'''
		if self.stepCount == 2:
			positions = self.hdlEnv.getPositions()
			self.startDistances, self.sumStartDist = self._computeDistances(positions)
		elif self.truncated:
			positions = self.hdlEnv.getPositions()
			endDistances, sumEndDist = self._computeDistances(positions)
			if not self.sumStartDist:
				self.sumStartDist = 1e-6
			progress = self.sumStartDist - sumEndDist
			self.score += 100*(progress / self.sumStartDist)
			with open('score.csv', 'a') as f:
				f.write(f"{round(self.score, 2)}\n")
			self.score = 0
		elif self.terminated:
			self.score = -11
			with open('score.csv', 'a') as f:
				f.write(f"{round(self.score, 2)}\n")
			self.score = 0

	def logScore(self):
		# log score
		if (self.calcReward.nearObjectID != self.calcReward.prevNearObjectID) and (self.calcReward.prevNearObjectID is not None):
			self.score += 1
			self.calcReward.positions = self.calcReward.handleEnv.getPositions()
			self.startDistance = self.calcReward.getDistObjToGoal(self.calcReward.nearObjectID)
		if self.stepCount == 2:
			self.calcReward.positions = self.calcReward.handleEnv.getPositions()
			self.startDistance = self.calcReward.getDistObjToGoal(self.calcReward.nearObjectID)
			print(f"Start distance: {self.startDistance}")
		elif self.truncated:
			if self.startDistance is None:
				self.startDistance = 0.0001
			self.calcReward.positions = self.calcReward.handleEnv.getPositions()
			print(f"Start distance: {self.startDistance}")
			print(f"ObjToGoal distance: {self.calcReward.getDistObjToGoal(self.calcReward.nearObjectID)}")
			self.score += (self.startDistance - self.calcReward.getDistObjToGoal(self.calcReward.nearObjectID)) / self.startDistance
			# safe score in csv file
			with open('score.csv', 'a') as f:
				f.write(f"{round(self.score, 2)}\n")
			self.score = 0
		elif self.terminated:
			self.score = -1
			with open('score.csv', 'a') as f:
				f.write(f"{round(self.score, 2)}\n")
			self.score = 0

	def step(self, action):
		self.hdlEnv.performAction(action)
		self.terminated = self.calcReward.taskFinished()
		if self.hdlEnv.checkMisbehaviour():
			self.terminated = True
			self.reward = -1000
		else:
			self.reward = self.calcReward.calcReward()
		if self.stepCount >= MAX_STEPS-1:
			self.truncated = True
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Terminated': self.terminated, 'Truncated': self.truncated}
		print(info)
		self.stepCount += 1
		observation = self.hdlEnv.getStates()
		#observation = self.calcReward.getStatePositions()
		
		#self.logScore()
		self.logScoreAllObjects()

		return observation, self.reward, self.terminated, self.truncated, info
	
	def reset(self, seed=None):
		super().reset(seed=seed)
		self.stepCount = 0
		self.terminated = False
		self.truncated  = False
		self.prevReward = 0
		self.hdlEnv.resetEnvironment()
		self.hdlEnv.robotToStartPose()
		self.hdlEnv.spawnGoals()
		self.hdlEnv.spawnObjects()
		self.calcReward.reset()
		
        # create observation
		observation = self.hdlEnv.getStates() # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
		#observation = self.calcReward.getStatePositions()

		info = {}

		print("SortingViaPushingEnv resetted")
		return observation, info
