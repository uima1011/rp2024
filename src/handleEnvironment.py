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
        self.dt = self.bullet_client.getPhysicsEngineParameters()['fixedTimeStep'] # timestep in seconds
        if not render:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)

    def getBoundaries(self):
        '''Returns the boundaries of the observation space.'''
        maxXRange = 1 # because relativ pos and offset because robot spawns behind table
        maxYRange = 1

        robotLow = np.array([-maxXRange, -maxYRange])
        robotHigh = np.array([maxXRange, maxYRange])
        objectLow = np.tile([-maxXRange, -maxYRange, -1], MAX_OBJECT_COUNT)
        objectHigh = np.tile([maxXRange, maxYRange, 1], MAX_OBJECT_COUNT)
        goalLow = np.tile([-maxXRange, -maxYRange, -1], len(colours))
        goalHigh = np.tile([maxXRange, maxYRange, 1], len(colours))

        lowBounds = np.concatenate([robotLow, objectLow, goalLow])
        highBounds = np.concatenate([robotHigh, objectHigh, goalHigh])
        return lowBounds, highBounds

    def resetEnvironment(self):
        self.bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.robot.home()
        self.IDs = {}
        # reset velocity
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

    def transformOriginToTableCenter(self, position):
        '''Transforms the origin to the center of the table and return the new position.'''
        x, y = position
        xCenterTransform = self.hO.tableCords['x'][0] + (self.hO.tableCords['x'][1] - self.hO.tableCords['x'][0]) / 2
        yCenterTransform = self.hO.tableCords['y'][0] + (self.hO.tableCords['y'][1] - self.hO.tableCords['y'][0]) / 2
        return x - xCenterTransform, y - yCenterTransform

    def normalize(self, value, mode):
        '''Normalize a value to the range [-1, 1] with respect to the axis'''
        if mode == 'relX': # use diagonal lenght of table
            xLength = self.hO.tableCords['x'][1] - self.hO.tableCords['x'][0] # + 2 * self.hO.tableOffset['x']
            normalizedValue = value / xLength
        elif mode == 'relY':
            yLength = self.hO.tableCords['y'][1] - self.hO.tableCords['y'][0] # + 2 * self.hO.tableOffset['y']
            normalizedValue = value / yLength
        elif mode == 'absX':
            normalizedValue = value / ((self.hO.tableCords['x'][1] - self.hO.tableCords['x'][0])/2+0.1)
        elif mode == 'absY':
            normalizedValue = value / ((self.hO.tableCords['y'][1] - self.hO.tableCords['y'][0])/2+0.1)
        elif mode == 'ang':
            wrappedAngle = np.arctan2(np.sin(value), np.cos(value)) # normalize angle to [-pi, pi]
            normalizedValue =  wrappedAngle / np.pi # normalize to [-1, 1]
        return np.clip(normalizedValue, -1, 1)

    def getRelativePositions(self):
        '''Returns the relative normalized positions and angle of objects and goals with respect to the robot's end-effector.'''
        objectStates, goalStates = [], []

        # Get robot position (end-effector pose)
        robotPosition = self.transformOriginToTableCenter(self.robot.get_eef_pose().translation[:2])
        robotState = [self.normalize(robotPosition[0], 'absX'), self.normalize(robotPosition[1], 'absY')]
        for key, ids in self.IDs.items():
            states = []
            for bulletID in ids:
                pos, ori = self.bullet_client.getBasePositionAndOrientation(bulletID)
                zAngle = R.from_quat(ori).as_euler('xyz')[2]

                # Compute relative position (object/goal position - robot position)
                transPos = self.transformOriginToTableCenter(pos[:2])
                relNormX = self.normalize(transPos[0], 'absX') # - robotPosition[0], 'relX')
                relNormY = self.normalize(transPos[1], 'absY') # - robotPosition[1], 'relY')
                zAngNorm = self.normalize(zAngle, 'ang') # no need for normalization since TCP is always at 0 deg
                relative_pos = [relNormX, relNormY, zAngNorm]
                states.extend(relative_pos)

            if 'goal' in key:
                goalStates.extend(states)
            else:
                objectStates.extend(states)
              

        # Pad object states to ensure consistent size
        paddedObjStates = np.pad(objectStates, (0, 3 * MAX_OBJECT_COUNT - len(objectStates)), constant_values=0)
        return robotState, paddedObjStates, goalStates

    def getStates(self):
        '''Combines positions and velocities into a single observation.'''
        # Get current positions
        robotState, paddedObjStates, goalStates = self.getRelativePositions()

        # Concatenate states and velocities for the final observation
        return np.concatenate([robotState, paddedObjStates, np.array(goalStates)])
        
    def getPositions(self):
        '''returns dict with nested list for dealing with position of robot, objects and goals individualy'''
        positionDict = {'goals': {}, 'objects': {}, 'robot': []}
        for key, ids in self.IDs.items():
            if 'goal' in key:
                positionDict['goals'][key] = {}
                for id in ids:
                    pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
                    zAngle = R.from_quat(ori).as_euler('xyz')[2]
                    positionDict['goals'][key][id] = [pos[0], pos[1], zAngle]
            else:
                positionDict['objects'][key] = {}
                for id in ids:
                    pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
                    zAngle = R.from_quat(ori).as_euler('xyz')[2]
                    positionDict['objects'][key][id] = [pos[0], pos[1], zAngle]
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
        [robotX, robotY] = self.robot.get_eef_pose().translation[:2]
        tableX = self.hO.tableCords['x']
        tableY = self.hO.tableCords['y']
        leaved = True
        if robotX < tableX[1]+self.hO.tableOffset['x'] and robotX > tableX[0]-self.hO.tableOffset['x']: # check x
            if robotY < tableY[1]+self.hO.tableOffset['y'] and robotY > tableY[0]-self.hO.tableOffset['y']: # check y
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
        self.tableOffset = {'x': 0.1, 'y': 0.1}
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
        self.positions = self.handleEnv.getPositions()
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj, self.nearObjectID  = None, None, None, None
        self.prevDistRobToObj, self.prevNearObjectID = self.getNearestObjToRob()
        self.prevDistObjToGoal, self.prevDistRobToGoal =  self.getDistObjToGoal(self.prevNearObjectID), self.getDistRobToGoal(self.prevNearObjectID)

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
        # Case 1: If there's no selected nearest object or the current object reached the goal
        if self.nearObjectID is None or self.checkObjectInsideGoal(self.nearObjectID):
            minDistance = float('inf')
            self.nearObjectID = None  # Reset nearest object ID
            
            # Iterate over all objects to find the nearest one
            for objPosDict in self.positions['objects'].values():
                for objID, objPosition in objPosDict.items():
                    distance = self.calculateDistance(self.positions['robot'], objPosition[:2])
                    if distance < minDistance and not self.checkObjectInsideGoal(objID):
                        minDistance = distance
                        self.nearObjectID = objID
            
            # If no valid object found, return None
            if self.nearObjectID is None:
                return None, None
            return minDistance, self.nearObjectID
        
        # Case 2: If the nearest object is already selected and valid
        for objDict in self.positions['objects'].values():
            if self.nearObjectID in objDict:
                obj_position = objDict[self.nearObjectID]
                dist = self.calculateDistance(self.positions['robot'], obj_position[:2])
                return dist, self.nearObjectID
        
        # Handle unexpected cases where the object ID is missing
        self.nearObjectID = None
        return None, None


    def getDistObjToGoal(self, objID):
        objName, objPos = next(((obj, pos[objID]) for (obj, pos) in self.positions['objects'].items() if objID in self.positions['objects'][obj]), (None, None))
        if objName is None or objPos is None:
            return float('inf')
        colour = objName.split('_')[1]
        _, goalPos = next(((obj, pos[1]) for (obj, pos) in self.positions['goals'].items() if f'goal_{colour}' in obj), (None, None))
        if goalPos is None:
            return float('inf')
        return self.calculateDistance(objPos[:2], goalPos[:2])
    
    def getDistRobToGoal(self, objID):
        objName, _ = next(((obj, pos[objID]) for (obj, pos) in self.positions['objects'].items() if objID in self.positions['objects'][obj]), (None, None))
        if objName is None:
            return float('inf')
        colour = objName.split('_')[1]
        _, goalPos = next(((obj, pos[1]) for (obj, pos) in self.positions['goals'].items() if f'goal_{colour}' in self.positions['goals']), (None, None))
        if goalPos is None:
            return float('inf')
        return self.calculateDistance(self.positions['robot'], goalPos[:2])

    
    def polarityCross(self, nearestObjID):
        """
        Determine if the robot's TCP is on the correct side of the object using the cross product method.
        
        Parameters:
        object_pos (list or tuple): Position of the object [x, y].
        goal_pos (list or tuple): Position of the goal [x, y].
        tcp_pos (list or tuple): Position of the TCP [x, y].
        
        Returns:
        int: +1 if on the correct side, -1 if on the wrong side, 0 if exactly on the line.
        """
        for key, objPositions in self.positions['objects'].items():
            if nearestObjID in objPositions:
                objPos = objPositions[nearestObjID][:2]
                colour = key.split('_')[1]
                break
        goalPos = next((pos[1] for (obj, pos) in self.positions['goals'].items() if f'goal_{colour}' in obj), None)[:2]
        
        # Extract coordinates

        # Compute vectors
        og = np.array(goalPos) - np.array(objPos)  # Vector from object to goal
        ot = np.array(self.positions['robot']) - np.array(objPos)   # Vector from object to TCP
        
        # Normalize vectors
        og_norm = np.linalg.norm(og)
        ot_norm = np.linalg.norm(ot)
        
        if og_norm == 0 or ot_norm == 0:
            raise ValueError("Zero-length vector encountered.")
        
        cos_theta = np.dot(og, ot) / (og_norm * ot_norm)  # Compute cosine of angle
        
        # Return polarity: +1 if cos_theta < 0, -1 otherwise
        return +1 if cos_theta < 0 else -1
        

    def checkAlignment(self, nearestObjID):
        for key, objPositions in self.positions['objects'].items():
            if nearestObjID in objPositions:
                objPos = objPositions[nearestObjID][:2]
                colour = key.split('_')[1]
                break
        goalPos = next((pos[1] for (obj, pos) in self.positions['goals'].items() if f'goal_{colour}' in obj), None)[:2]
        
        vec_obj_goal = np.array(goalPos) - np.array(objPos)
        vec_rob_obj = np.array(objPos) - np.array(self.positions['robot'])
        
        # Normalize vectors
        vec_obj_goal /= np.linalg.norm(vec_obj_goal)
        vec_rob_obj /= np.linalg.norm(vec_rob_obj)
        
        # Dot product gives cosine of angle between vectors
        alignment = np.dot(vec_obj_goal, vec_rob_obj)
        
        # Reward based on alignment (closer to 1 is better)
        print(f"Alignment: {alignment}")
        return alignment

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

        print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions['objects'].items() if self.nearObjectID in self.positions['objects'][obj]), None))

        
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
        print(f"Nearest Object:", next(((obj, pos[self.nearObjectID]) for (obj, pos) in self.positions['objects'].items() if self.nearObjectID in self.positions['objects'][obj]), None))
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
        if self.distRobToObj is None:
            return -0.5 # obj in goal at first step? TODO check
        distCloseToRobot = 0.05
        distCloseToGoal = self.handleEnv.hO.goalWidths['x']/2-self.handleEnv.hO.objectWidth/2
        if self.distRobToObj > distCloseToRobot: # Task: move to object
            print("task move robot to object")
            reward = self.prevDistRobToObj - self.distRobToObj
        elif self.distObjToGoal > distCloseToGoal: # Task: push object into goal
            print("task push object into goal")
            reward = self.prevDistObjToGoal - self.distObjToGoal
        return reward

    def calcReward4(self):
        self.positions = self.handleEnv.getPositions()
        self.distRobToObj, self.nearObjectID = self.getNearestObjToRob()
        self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
        self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
        if self.prevNearObjectID is None and self.nearObjectID is None: # object spawned in goal
            reward = 0
            return self.normReward(reward)
        if self.prevNearObjectID != self.nearObjectID and self.prevNearObjectID is not None: 
            reward = 100
            self.prevNearObjectID = self.nearObjectID
            return self.normReward(reward)
        
        distCloseToRobot = 0.05
        distCloseToGoal = self.handleEnv.hO.goalWidths['x']/2-self.handleEnv.hO.objectWidth/2
        reward = 0
        # Polarity check (penalty if on wrong side)
        side = self.polarityCross(self.nearObjectID)
        if side < 0:
            reward -= 5

        if self.distRobToObj > distCloseToRobot: # Task: move to object
            print("task move robot to object")
            reward += (self.prevDistRobToObj - self.distRobToObj)*5
        elif self.distObjToGoal > distCloseToGoal: # Task: push object into goal
            print("task push object into goal")
            reward += (self.prevDistObjToGoal - self.distObjToGoal)*5
            alignmentReward = self.checkAlignment(self.nearObjectID)
            reward += alignmentReward*10
        else:
            print("Error, shouldnt come here")
            reward = 0

        reward -= 0.1 # Time based penalty 
        # Prevent oscillations
        if self.distRobToObj > self.prevDistRobToObj:
            reward -= 1  # Penalty for increasing distance to object   
        self.prevNearObjectID = self.nearObjectID
        self.prevDistRobToObj = self.distRobToObj
        self.prevDistObjToGoal = self.distObjToGoal
        self.prevDistRobToGoal = self.distRobToGoal
        return self.normReward(reward)

    def normReward(self, reward):
        min_reward = -1000
        max_reward = 1000

        # Normalize reward to [-1, 1]
        normalized_reward = 2 * (reward - min_reward) / (max_reward - min_reward) - 1
        return normalized_reward

    def taskFinished(self):
        '''checks if all objects are inside their goal zones --> returns true otherwhise false'''
        for key, values in self.handleEnv.IDs.items():
            if 'goal' not in key and 'robot' not in key:
                for id in values:
                    if not self.checkObjectInsideGoal(id):
                        return False
        return True

    def taskTimeout(self, steps, episode, prevEpisode, counter):
        '''check if current Task timeouts with increasing complexity and time'''
        maxSteps = 1000
        if episode%50==0 and episode != prevEpisode:
            prevEpisode = episode
            counter += 1
        
        if steps >= maxSteps: # steps >= 50+10*counter or 
            timeout = True
        else:
            timeout = False
        return timeout, counter, prevEpisode
        
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
