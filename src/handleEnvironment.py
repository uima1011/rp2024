from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
import pybullet as p

from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import time

colours = ['green'] # ['green', 'red']
objectFolders = ['cubes'] # ['signs', 'cubes']
parts = ['cube'] # ['plus', 'cube']
MAX_OBJECT_COUNT = 2*len(colours)*len(parts)

class HandleEnvironment():
    def __init__(self, render, assets_folder):
        self.urdfPathRobot = os.path.join(assets_folder, 'urdf', 'robot_without_gripper.urdf')
        self.urdfPathGoal = os.path.join(assets_folder, 'objects', 'goals')
        # init objects
        self.hO = HandleObjects(assets_folder)
        self.IDs = {}
        self.nearestObjID = None
        # robot and sim
        self.bullet_client = BulletClient(connection_mode=p.GUI)
        self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Initialize previous states for velocity computation
        self.previous_object_states = None
        self.previous_robot_state = None
        self.dt = self.bullet_client.getPhysicsEngineParameters()['fixedTimeStep'] # timestep in seconds
        if not render:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)

    def getBoundaries(self):
        '''Returns the boundaries of the observation space.'''
        maxXRange = self.hO.tableCords['x'][1] - self.hO.tableCords['x'][0] + 0.1 # because relativ pos and offset because robot spawns behind table
        maxYRange = self.hO.tableCords['y'][1] - self.hO.tableCords['y'][0] + 0.1

        robotLow = np.array([-maxXRange, -maxYRange])
        robotHigh = np.array([maxXRange, maxYRange])
        objectLow = np.tile([-maxXRange, -maxYRange, -2*np.pi], MAX_OBJECT_COUNT)
        objectHigh = np.tile([maxXRange, maxYRange, 2*np.pi], MAX_OBJECT_COUNT)
        goalLow = np.tile([-maxXRange, -maxYRange, -2*np.pi], len(colours))
        goalHigh = np.tile([maxXRange, maxYRange, 2*np.pi], len(colours))

        lowBounds = np.concatenate([robotLow, objectLow, goalLow])
        highBounds = np.concatenate([robotHigh, objectHigh, goalHigh])
        return lowBounds, highBounds

    def resetEnvironment(self):
        self.bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.robot.home()
        self.IDs = {}
        # reset velocity
        self.previous_object_states = None
        self.previous_robot_state = None
        self.hO.reset() # reset objects and goals
        print("Environment resetted")

    def robotToStartPose(self):
        target_pose = Affine(translation=[0.25, -0.34, -0.1], rotation=[-np.pi, 0, np.pi/2])
        self.robot.lin(target_pose)

    def spawnGoals(self):
        goals = self.hO.generateGoals()
        if goals != None:
            for val, col in zip(goals.values(), colours):
                urdfPath = os.path.join(self.urdfPathGoal, f'goal_{col}.urdf')
                self.IDs[f'goal_{col}'] = []
                objID = self.bullet_client.loadURDF(
                                urdfPath,
                                val['pose'].translation,
                                val['pose'].quat,
                                flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                            )
                self.IDs[f'goal_{col}'].append(objID)

    def spawnObjects(self):
        objects = self.hO.generateObjects()
        if objects is not None:
            for key, vals in objects.items():
                self.IDs[key] = []
                for pose in vals['poses']: # spawn all objects of same type and colour
                    objID = self.bullet_client.loadURDF(
                                    vals['urdfPath'],
                                    pose.translation,
                                    pose.quat,
                                    flags=self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                                )
                    self.IDs[key].append(objID)
            # Let objects settle
            for _ in range(100):
                self.bullet_client.stepSimulation()
                time.sleep(1/100)

    def getIDs(self):
        return self.IDs          

    def _getCoordinates(self):
        '''Returns the relative coordinates of objects and goals with respect to the robot's end-effector.'''
        objectStates, goalStates = [], []

        # Get robot position (end-effector pose)
        robotState = self.robot.get_eef_pose().translation[:2]

        for key, ids in self.IDs.items():
            states = []
            for id in ids:
                pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
                zAngle = R.from_quat(ori).as_euler('xyz')[2]

                # Compute relative position (object/goal position - robot position)
                relative_pos = [pos[0] - robotState[0], pos[1] - robotState[1], zAngle]
                states.extend(relative_pos)

            if 'goal' in key:
                goalStates.extend(states)
            else:
                objectStates.extend(states)
              

        # Pad object states to ensure consistent size
        paddedObjStates = np.pad(objectStates, (0, 3 * MAX_OBJECT_COUNT - len(objectStates)), constant_values=0)

        # Store current states for velocity computation
        self.previous_object_states = paddedObjStates
        self.previous_robot_state = robotState

        return robotState, paddedObjStates, goalStates

    def getStates(self):
        '''Combines positions and velocities into a single observation.'''
        # Get current positions
        robotState, paddedObjStates, goalStates = self._getCoordinates()

        # Concatenate states and velocities for the final observation
        return np.concatenate([robotState, paddedObjStates, np.array(goalStates)])
    
    def getStatesOnlyNearestObject(self, nearestObjID):
        self.nearestObjID = nearestObjID
        '''Only returns robot state, nearest object rel and rel goals'''
        # Get current positions
        robotState, paddedObjStates, goalStates = self._getCoordinates()

        # Compute velocities based on current and previous positions
        robotVelocity, paddedObjVelocities = self._getVelocities(robotState, paddedObjStates)

        # Concatenate states and velocities for the final observation
        return np.concatenate([robotState, np.zeros(2), paddedObjStates, np.zeros(3 * MAX_OBJECT_COUNT), np.array(goalStates)])


    def _getVelocities(self, current_robot_state, current_object_states):
        '''Computes the velocities of the robot and objects using finite differences.'''
        # Ensure previous states are available
        if self.previous_object_states is None or self.previous_robot_state is None:
            zero_robot_velocity = np.zeros(2)
            zero_object_velocity = np.zeros(3 * MAX_OBJECT_COUNT)
            return zero_robot_velocity, zero_object_velocity

        # Compute finite differences for velocities
        robotVelocity = (np.array(current_robot_state) - np.array(self.previous_robot_state)) / self.dt
        objectVelocities = (np.array(current_object_states) - np.array(self.previous_object_states)) / self.dt

        # Pad object velocities to ensure consistent size
        paddedObjVelocities = np.pad(objectVelocities, (0, 3 * MAX_OBJECT_COUNT - len(objectVelocities)), constant_values=0)

        return robotVelocity, paddedObjVelocities
    
    def getPositions(self):
        '''returns dict with nested list for dealing with position of robot, objects and goals individualy'''
        positionDict = {}
        for key, ids in self.IDs.items():
            positionDict[key] = {}
            for id in ids:
                pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
                zAngle = R.from_quat(ori).as_euler('xyz')[2]
                positionDict[key][id] = [pos[0], pos[1], zAngle]
        positionDict['robot'] = self.robot.get_eef_pose().translation[:2]
        return positionDict

    def performAction(self, action):
        current_pose = self.robot.get_eef_pose()
        fixed_orientation = [-np.pi, 0, np.pi/2]

        if action == 0:  # Move left
            new_x = current_pose.translation[0] - 0.01
            new_y = current_pose.translation[1]
        elif action == 1:  # Move right
            new_x = current_pose.translation[0] + 0.01
            new_y = current_pose.translation[1]
        elif action == 2:  # Move forward
            new_x = current_pose.translation[0]
            new_y = current_pose.translation[1] + 0.01
        elif action == 3:  # Move backward  
            new_x = current_pose.translation[0]
            new_y = current_pose.translation[1] - 0.01
        else:
            return  # Invalid action

        # Create a new target pose with updated x and y, keeping z the same and setting the fixed orientation
        target_pose = Affine(
            translation=[new_x, new_y, -0.1],  # Keep z the same
            rotation=fixed_orientation  # Set the fixed orientation
        )
        self.robot.lin(target_pose)   

    def robotLeavedWorkArea(self):
        '''returns True if robot out of Area''' # TODO
        offX = 0.25 # offset
        offY = 0.1
        [robotX, robotY] = self.robot.get_eef_pose().translation[:2]
        tableX = self.hO.tableCords['x']
        tableY = self.hO.tableCords['y']
        leaved = True
        if robotX < tableX[1]+offX and robotX > tableX[0]-offX: # check x
            if robotY < tableY[1]+offY and robotY > tableY[0]-offY: # check y
                leaved = False
                
        if leaved:
            print(f"Robot leaved work area")
        return leaved


    def objectOffTable(self):
        for key , values in self.IDs.items():
            if 'goal' not in key and 'robot' not in key:
                for id in values:
                    pos,_ = self.bullet_client.getBasePositionAndOrientation(id)
                    z = pos[2]
                    if z < 0: 
                        print(f"Error: Object {key} with ID {id} is off the table")
                        return True
        return False

    def checkMisbehaviour(self):
        '''check behaviour of robot and objects and return true if something misbehaves'''
        misbehaviour = self.objectOffTable() | self.robotLeavedWorkArea()
        if misbehaviour==True:
            print(f"Misbehaviour: {misbehaviour}")
        return misbehaviour
    
    def close(self):
        p.disconnect()
        
class HandleObjects():
    def __init__(self, assets_folder):
        self.tableCords = {
                    'x':[0.3, 0.9], # min, max
                    'y':[-0.29, 0.29]
                    }
        self.objectWidth = 0.05
        self.goalWidths = {
                            'x': 3*self.objectWidth + 0.01, # numb*min_size_object + offset --> 9 objets fit in goal
                            'y': 3*self.objectWidth + 0.01
                            }  

        self.goals = {}
        self.objects = {f'{part}_{colour}': {'poses': [], 'urdfPath': None} for part in parts for colour in colours}
        self.urdfPathObjects = os.path.join(assets_folder, 'objects')

    # Objects:
    def check_collision(self, position_to_check, other_positions, min_safety_distance = 0.1):
        '''Check if a position is too close to any existing positions'''
        for other_position in other_positions:
            distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(position_to_check[:2], other_position[:2])))  # Only check x,y
            if distance < min_safety_distance:
                return True
        return False

    def generate_valid_position(self, existing_positions, min_safety_distance=0.1, max_attempts=100):
        '''Generate a random position that doesn't collide with existing objects'''
        for _ in range(max_attempts):
            x = np.random.uniform(self.tableCords['x'][0], self.tableCords['x'][1])
            y = np.random.uniform(self.tableCords['y'][0], self.tableCords['y'][1])
            z = 0.1
            if not self.check_collision([x, y, z], existing_positions, min_safety_distance):
                return [x, y, z]
        return None
    
    def generate_single_object(self, existing_positions):
        """Spawn a single object at a random position avoiding collisions"""
        position = self.generate_valid_position(existing_positions)
        if position == None:
            return None
        existing_positions.append(position) #add to list of positions
        angle = np.random.uniform(0, 2*np.pi)
        objectPose = Affine(translation = position, rotation=(0, 0, angle))
        return objectPose
    
    def generateObjects(self, max_attempts=100):
        existing_positions = []
        for folder, part in zip(objectFolders, parts):
            for col in colours:
                urdfPath = os.path.join(self.urdfPathObjects, folder, f'{part}_{col}.urdf')
                objCount = np.random.randint(1, (MAX_OBJECT_COUNT / len(colours) / len(parts))+1)
                spawned_count = 0
                for _ in range(objCount):
                    for _ in range(max_attempts):
                        objectPose = self.generate_single_object(existing_positions)
                        self.objects[f'{part}_{col}']['urdfPath'] = urdfPath
                        if objectPose is not None:
                            self.objects[f'{part}_{col}']['poses'].append(objectPose)
                            spawned_count += 1
                            break
                    else:
                        print(f'Warning: Could not generate object {part}_{col} after {max_attempts} attempts')
                if spawned_count < objCount:
                    print(f'Warning: Could not generate all objects for {part}_{col}')
        if len(self.objects) == 0:
            print('Error: Could not generate any objects')
            return None
        else:
            return self.objects
        
    def get_state_obj_z(self):
        object_z_positions = {}
        print(f"Self.objects: {self.objects}")
        return object_z_positions

    
    # Goals:
    def generate_single_goal_area(self, table_coords, goal_width):
        """Generate coordinates for a single goal area inside the table with respect to the width of the rectangle"""
        x_goal_min = np.random.uniform(table_coords['x'][0], table_coords['x'][1] - goal_width['x'])
        y_goal_min = np.random.uniform(table_coords['y'][0], table_coords['y'][1] - goal_width['y'])
        x_goal_max = x_goal_min + goal_width['x']
        y_goal_max = y_goal_min + goal_width['y']
        return (x_goal_min, y_goal_min, x_goal_max, y_goal_max)
    
    def check_rectangle_overlap(self, rect1, rect2):
        """Check if two rectangles overlap"""
        if rect1[2] < rect2[0] or rect2[2] < rect1[0]:  # If one rectangle is on the left side of the other
            return False
        if rect1[3] < rect2[1] or rect2[3] < rect1[1]:  # If one rectangle is above the other
            return False
        return True
    
    def generate_goal_pose(self, goal_coords, z_goal):
        """Generates pose for a goal area with random rotation arround z axis"""
        x_goal_min, y_goal_min, x_goal_max, y_goal_max = goal_coords
        mid_x = (x_goal_max - x_goal_min) / 2 + x_goal_min
        mid_y = (y_goal_max - y_goal_min) / 2 + y_goal_min
        translation = [mid_x, mid_y, z_goal]
        rotation = [0, 0, np.random.uniform(0, 2*np.pi)]  # rotate random around z axis
        goalPose = Affine(translation, rotation)
        return goalPose
    
    def generateGoals(self, z_goal = -0.01, max_attempts=100):
        '''Generate two non-overlapping goal areas. Returns False if unable to generate non-overlapping areas after max_attempts.'''
        for _ in range(max_attempts):
            goal1_coords = self.generate_single_goal_area(self.tableCords, self.goalWidths) # generate goal area
            goal2_coords = self.generate_single_goal_area(self.tableCords, self.goalWidths) 
            if not self.check_rectangle_overlap(goal1_coords, goal2_coords):
                goal1_pose = self.generate_goal_pose(goal1_coords, z_goal) # generates pose for goal area center
                goal2_pose = self.generate_goal_pose(goal2_coords, z_goal)
                self.goals = {'1': {'pose': goal1_pose, 'coords': goal1_coords}, '2': {'pose': goal2_pose, 'coords': goal2_coords}}
                return self.goals
        print(f"Unable to generate non-overlapping goal areas after {max_attempts} attempts.")        
        self.goals = {}
        return None
    
    def reset(self):
        self.goals = {}
        self.objects = {f'{part}_{colour}': {'poses': [], 'urdfPath': None} for part in parts for colour in colours}
        print("Objects and goals resetted")

class CalcReward():
    def __init__(self, handleEnv):
        self.handleEnv = handleEnv
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj  = None, None, None
        self.prevDistRobToGoal, self.prevDistObjToGoal, self.prevDistRobToObj = None, None, None
        self.nearObjectID, self.prevNearObjectID = None, None
        self.positions = self.handleEnv.getPositions()

    def reset(self):
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj  = None, None, None
        self.prevDistRobToGoal, self.prevDistObjToGoal, self.prevDistRobToObj = None, None, None
        self.nearObjectID, self.prevNearObjectID = None, None
        self.positions = self.handleEnv.getPositions()
        print("Reward calculator resetted")

    def calculateDistance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def checkObjectInsideGoal(self, objID):
        distDefInsideGoal = self.handleEnv.hO.goalWidths['x']/2-self.handleEnv.hO.objectWidth/2
        if self.getDistObjToGoal(objID) < distDefInsideGoal:
            return True
        else: # outside goal
            return False
        
    def getNearestObjToRob(self):
        while(True):
            if self.nearObjectID is None: # first or last step
                minDistance = float('inf')
                for key, positionsDict in self.positions.items():
                    if 'robot' not in key and 'goal' not in key:  # We don't want to compare robot to itself or to goal
                        # Check each position for an object (in case of multiple positions like 'plus_red')
                        for id, obj_position in positionsDict.items():
                            distance = self.calculateDistance(self.positions['robot'], obj_position[:2])
                            if distance < minDistance: # new minDistance and objekt outside of goal
                                if not self.checkObjectInsideGoal(id):
                                    minDistance = distance
                                    self.nearObjectID = id
                if self.nearObjectID is None:
                    return None, None        
                return minDistance, self.nearObjectID
            else:
                if self.checkObjectInsideGoal(self.nearObjectID): # check if nearest object is inside goal area
                        self.nearObjectID = None
                        dist = None
                else:  # Abstand berechnen
                    for key, positionsDict in self.positions.items():
                        if self.nearObjectID in positionsDict:
                            dist = self.calculateDistance(self.positions['robot'], positionsDict[self.nearObjectID][:2])
                            break
                    return dist, self.nearObjectID    


    def getDistObjToGoal(self, objID):
        objName, objPos = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), (None, None))
        if objName is None or objPos is None:
            return float('inf')
        colour = objName.split('_')[1]
        _, goalPosDict = next(((obj, pos) for (obj, pos) in self.positions.items() if f'goal_{colour}' in obj), None)
        if goalPosDict is None:
            return float('inf')
        goalPos, = goalPosDict.values()
        return self.calculateDistance(objPos[:2], goalPos[:2])
    
    def getDistRobToGoal(self, objID):
        objName, _ = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), (None, None))
        if objName is None:
            return float('inf')
        colour = objName.split('_')[1]
        _, goalPos = next(((obj, pos[1]) for (obj, pos) in self.positions.items() if f'goal_{colour}' in self.positions), (None, None))
        if goalPos is None:
            return float('inf')
        return self.calculateDistance(self.positions['robot'], goalPos[:2])

    def calcReward(self):
        self.positions = self.handleEnv.getPositions()
        self.prevNearObjectID = self.nearObjectID
        # dictance robot to nearest object 
        self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
        if (self.nearObjectID != self.prevNearObjectID):
            # set previous distance to new nearest obj
            self.prevDistRobToObj = self.distRobToObj
            self.prevDistObjToGoal = self.getDistObjToGoal(self.nearObjectID)
            self.prevDistRobToGoal = self.getDistRobToGoal(self.nearObjectID)
            reward = 15
            return reward
        # distance of that object to its goal
        self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
        #distance of robot to goal for nearest object
        self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
        #remeber distances for next step
        reward = 0
        if (self.prevDistRobToObj - self.distRobToObj) > 0.005:
            reward += 0.9
        elif (self.distRobToObj - self.prevDistRobToObj) > 0.005:
            reward -= 0.9
        if (self.prevDistObjToGoal - self.distObjToGoal) > 0.005:
            reward += 5
        elif (self.distObjToGoal - self.prevDistObjToGoal) > 0.005:
            reward -= 0
        if (self.prevDistRobToGoal - self.distRobToGoal) > 0.005:
            reward -= 0.1
        elif (self.distRobToGoal - self.prevDistRobToGoal) > 0.005:
            reward += 0.1

        print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions.items() if self.nearObjectID in self.positions[obj]), None))

        
        self.prevDistRobToObj = self.distRobToObj
        self.prevDistObjToGoal = self.distObjToGoal
        self.prevDistRobToGoal = self.distRobToGoal

        return reward     
    
    def calcReward2(self): # use euclidian distance and reward pushing object into goal, punish switching objects
        reward = -0.5
        self.positions = self.handleEnv.getPositions()
        self.prevNearObjectID = self.nearObjectID
        self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
        self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
        self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
        if self.nearObjectID != self.prevNearObjectID: # new object --> reset treshhold so euclidian reward starts with 0
            self.prevDistRobToObj = self.distRobToObj
            self.prevDistObjToGoal = self.distObjToGoal
            self.prevDistRobToGoal = self.distRobToGoal
            reward =+ 15 # award one more object in goal
        if self.nearObjectID is not None: # valid previous distances
            rewardRobToObj = self.prevDistRobToObj - self.distRobToObj
            rewardObjToGoal = self.prevDistObjToGoal - self.distObjToGoal
            rewardRobToGoal = self.prevDistRobToGoal - self.distRobToGoal
        else: # objekt spawned in goal
            rewardRobToObj = 5 # equals same award as if object spawned in goal
            rewardObjToGoal = 0
            rewardRobToGoal = 0
        self.prevDistRobToObj = self.distRobToObj
        self.prevDistObjToGoal = self.distObjToGoal
        self.prevDistRobToGoal = self.distRobToGoal
        print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions.items() if self.nearObjectID in self.positions[obj]), None))
        return reward + (200*rewardRobToObj + 100*rewardObjToGoal - 50*rewardRobToGoal) # base reward + euclidian rewards
    
    def calcReward3(self):
        self.prevNearObjectID = self.nearObjectID
        self.positions = self.handleEnv.getPositions() #update positions to get net coords
        self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
        self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
        self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
        if self.nearObjectID != self.prevNearObjectID: # new object --> reset treshhold so euclidian reward starts with 0, return reward based if first or last for that object
            self.prevDistRobToObj = self.distRobToObj
            self.prevDistObjToGoal = self.distObjToGoal
            self.prevDistRobToGoal = self.distRobToGoal
            if self.prevNearObjectID == None: # first step
                reward = -0.5
            else:
                reward = 100
            return reward
        distCloseToRobot = 0.05
        distCloseToGoal = self.handleEnv.hO.goalWidths['x']/2-self.handleEnv.hO.objectWidth/2
        if self.distRobToObj > distCloseToRobot: # Task: move to object
            print("task move robot to object")
            reward = self.prevDistRobToObj - self.distRobToObj
        elif self.distObjToGoal > distCloseToGoal: # Task: push object into goal
            print("task push object into goal")
            reward = self.prevDistObjToGoal - self.distObjToGoal
        return reward



    def taskFinished(self):
        '''checks if all objects are inside their goal zones --> returns true otherwhise false'''
        for key, values in self.handleEnv.IDs.items():
            if 'goal' not in key and 'robot' not in key:
                for id in values:
                    if not self.checkObjectInsideGoal(id):
                        return False
        return True

    def taskTimeout(self, steps, episode, counter):
        '''check if current Task timeouts with increasing complexity and time'''
        if episode%50==0:
            counter += 1
        
        if steps >= 50*counter:
            timeout = True
        else:
            timeout = False
        return timeout, counter
        # maxDistRobToGoal = 3*0.05 + 0.02 # half goal width + offset
        # maxDistObjToGoal = 3*0.05 + 0.01 # half goal width + offset
        # maxDistRobToObj = 0.05/2 + 0.01 # half object width + offset
        # if self.distRobToGoal is None or self.distObjToGoal is None or self.distRobToObj is None:
        #     return False
        # elif steps>=500 and self.distRobToGoal > maxDistRobToGoal: # go to goal
        #     print(f"Timeout: robot too far away from goal")
        #     timeout = True
        # elif steps>=500 and self.distObjToGoal > maxDistObjToGoal:
        #     print(f"Timeout: object too far away from goal")
        #     timeout = True
        # if episode < 30:
        #     if steps>=100 and self.distRobToObj > maxDistRobToObj: # go to object
        #         print(f"Timeout: robot too far away from nearest object")
        #         timeout = True
        #     else:
        #         timeout = False
        # elif episode < 10:
        #     if steps>=50 and self.distRobToObj > maxDistRobToObj: # go to object
        #         print(f"Timeout: robot too far away from nearest object")
        #         timeout = True
        #     else:
        #         timeout = False
        # else:
        #     if steps>=400 and self.distRobToObj > maxDistRobToObj: # go to object
        #         print(f"Timeout: robot too far away from nearest object")
        #         timeout = True
        #     else:
        #         timeout = False
        # if steps>=70 and self.distRobToObj > maxDistRobToObj: # go to object
        #     print(f"Timeout: robot too far away from nearest object")
        #     timeout = True
        # else:
        #     timeout = False
        # return timeout, counter

def main():
    hEnv = HandleEnvironment(render=True, assets_folder="/home/group1/workspace/assets")
    hEnv.spawnGoals()
    hEnv.spawnObjects()

    state_obj_z = hEnv.objectOffTable()
    print(f"State object z: {state_obj_z}")

    hEnv.robotToStartPose()
    calcRew = CalcReward(hEnv)

    # ids = hEnv.getIDs()
    # pprint(ids)
  
    # # input('Press Enter to continue')
    # states = hEnv.getStates()
    # print('States:')
    # pprint(states)

    # input('Press Enter to continue')

if __name__ == '__main__':
    main()
