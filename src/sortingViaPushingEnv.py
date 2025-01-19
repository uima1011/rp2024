import gymnasium as gym
import numpy as np

from handleEnvironment import HandleEnvironment, CalcReward

# Setup Simulation
RENDER = True
ASSETS_PATH = "/home/group1/workspace/assets"

# Train:
MAX_STEPS = 200

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
		#state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * MAX_OBJECT_COUNT + GOAL_STATE_COUNT * GOAL_COUNT # robot + max objects + goal states
		state_dim = ROBOT_STATE_COUNT + OBJECT_STATE_COUNT * 1 + GOAL_STATE_COUNT * 1 # robot + max objects + goal states
		self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
											shape=(state_dim,), dtype=np.float64)
		self.hdlEnv = HandleEnvironment(RENDER, ASSETS_PATH)
		self.calcReward = CalcReward(self.hdlEnv)
		self.stepCount = 0
		self.startDistance = None
		self.score = 0
		
	def step(self, action):
		self.hdlEnv.performAction(action)
		self.terminated = self.calcReward.taskFinished()
		if self.hdlEnv.checkMisbehaviour():
			self.terminated = True
			self. reward = -1000
		else:
			self.reward = self.calcReward.calcReward()
		if self.stepCount >= MAX_STEPS-1:
			self.truncated = True
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Terminated': self.terminated, 'Truncated': self.truncated}
		print(info)
		self.stepCount += 1
		#observation = self.hdlEnv.getStates()
		observation = self.calcReward.getStatePositions()
		
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
		#observation = self.hdlEnv.getStates() # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ)
		observation = self.calcReward.getStatePositions()

		info = {}

		print("SortingViaPushingEnv resetted")
		return observation, info
