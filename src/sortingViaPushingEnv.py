import gymnasium as gym
import numpy as np

from handleEnvironment import HandleEnvironment, CalcReward

# Setup Simulation
RENDER = True
ASSETS_PATH = "/home/group1/workspace/assets"

# Train:
MAX_STEPS = 1000

# Environment
colours = ['green', 'red']
objectFolders = ['signs', 'cubes']
parts = ['plus', 'cube']

ROBOT_STATE_COUNT = 4 # x and y pos + vel
MAX_OBJECT_COUNT = 4*len(colours)*len(parts) # max 4 of each object type and colour
GOAL_COUNT = len(colours) # red and green
OBJECT_STATE_COUNT = 6 # x, y and rotation arround z (pos + vel)
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
		self.reward, nearestObjID = self.calcReward.calcReward() # TODO test
		self.done = self.hdlEnv.checkMisbehaviour() # TODO 
		if self.stepCount >= MAX_STEPS-1:
			self.truncated = True
		else:
			self.truncated = False # TODO ist das nicht sowieso schon false?
		info = {'Step': self.stepCount, 'Reward': self.reward, 'Action': action, 'Done': self.done, 'Truncated': self.truncated}
		print(info)
		self.stepCount += 1
		observation = self.hdlEnv.getStatesOnlyNearestObject(nearestObjID) # TODO test
		print(f'nearestObjID: {nearestObjID}')
		print(observation)
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
		_, nearestObjID = self.calcReward.calcReward() # TODO test
		self.calcReward.reset()
		
        # create observation
		observation = self.hdlEnv.getStatesOnlyNearestObject(nearestObjID) # robot state, object state, goal state (x,y|x,y,degZ|x,y,degZ) TODO test
		info = {}

		print("SortingViaPushingEnv resetted")
		return observation, info
