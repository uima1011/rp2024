from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
import pybullet as p

from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import time
from pprint import pprint

colours = ['green', 'red']
objectFolders = ['signs', 'cubes']
parts = ['plus', 'cube']
MAX_OBJECT_COUNT = 4*len(colours)*len(parts)

class HandleEnvironment():
    def __init__(self, render, assets_folder):
        self.urdfPathRobot = os.path.join(assets_folder, 'urdf', 'robot_without_gripper.urdf')
        self.urdfPathGoal = os.path.join(assets_folder, 'objects', 'goals')
        self.hO = HandleObjects(assets_folder)
        self.IDs = {}
        self.bullet_client = BulletClient(connection_mode=p.GUI)
        self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if not render:
            self.bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.robot.home()

    def resetEnvironment(self):
        self.bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=self.bullet_client, urdf_path=self.urdfPathRobot)
        self.robot.home()
        self.IDs = {}
        self.hO.reset() # reset objects and goals

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

    def getStates(self):
        '''returns flattened list as observation for robot, objects and goals'''
        objectStates, goalStates = [], []
        for key, ids in self.IDs.items():
            states = []
            for id in ids:
                pos, ori = self.bullet_client.getBasePositionAndOrientation(id)
                zAngle = R.from_quat(ori).as_euler('xyz')[2]
                states.extend([pos[0], pos[1], zAngle])
            if 'goal' in key:
                goalStates.extend(states)
            else:
                objectStates.extend(states)
        robotState = self.robot.get_eef_pose().translation[:2]
        paddedObjStates = np.pad(objectStates, (0, 3*MAX_OBJECT_COUNT-len(objectStates)), constant_values=0)
        return np.concatenate([robotState, paddedObjStates, np.array(goalStates)])

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
    
    def objectOffTable(self):
        '''checks objects positions z height and returns true if its under the table''' # TODO
        return False

    def robotLeavedWorkArea(self):
        '''returns True if robot out of Area''' # TODO
        [robotX, robotY] = self.robot.get_eef_pose().translation[:2]
        tableX = self.hO.tableCords['x']
        tableY = self.hO.tableCords['y']
        leaved = True
        if robotX < tableX[1] and robotX > tableX[0]: # check x
            if robotY < tableY[1] and robotY > tableY[0]: # check y
                leaved = False
        return False # TODO activate with returning leaved
    
    def checkMisbehaviour(self):
        '''check behaviour of robot and objects and return true if something misbehaves'''
        misbehaviour = self.objectOffTable() | self.robotLeavedWorkArea()
        return misbehaviour

class HandleObjects():
    def __init__(self, assets_folder):
        self.tableCords = {
                    'x':[0.3, 0.9], # min, max
                    'y':[-0.29, 0.29]
                    }
        self.goal_width = {
                            'x': 0.15 + 0.01, # min_size_object + offset
                            'y': 0.15 + 0.01
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
                objCount = np.random.randint(1, MAX_OBJECT_COUNT / len(colours) / len(parts))
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
            print("Some Objects failed to generate")
            return self.objects

    
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
            goal1_coords = self.generate_single_goal_area(self.tableCords, self.goal_width) # generate goal area
            goal2_coords = self.generate_single_goal_area(self.tableCords, self.goal_width) 
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


class CalcReward():
    def __init__(self, handleEnv):
        self.handleEnv = handleEnv
        self.distRobToGoal, self.distObjToGoal, self.distRobToObj  = None, None, None
        self.prevDistRobToGoal, self.prevDistObjToGoal, self.prevDistRobToObj = None, None, None
        self.nearObjectID, self.prevNearObjectID = None, None
        self.positions = self.handleEnv.getPositions()

    def calculateDistance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def checkObjectInsideGoal(self, objID):
        distDefInsideGoal = 0.05
        if self.getDistObjToGoal(objID) < distDefInsideGoal:
            return True
        else: # outside goal
            return False
        
    def getNearestObjectToRobot(self):
        minDistance = float('inf')
        for key, positionsDict in self.positions.items():
            if 'robot' not in key and 'goal' not in key:  # We don't want to compare robot to itself or to goal
                # Check each position for an object (in case of multiple positions like 'plus_red')
                for id, obj_position in positionsDict.items():
                    distance = self.calculateDistance(self.positions['robot'], obj_position[:2])
                    if distance < minDistance and not self.checkObjectInsideGoal(id): # new minDistance and objekt outside of goal
                        minDistance = distance
                        closestObjectID = id
        return minDistance, closestObjectID

    def getDistObjToGoal(self, objID):
        objName, objPos = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), None)
        colour = objName.split('_')[1]
        _, goalPosDict = next(((obj, pos) for (obj, pos) in self.positions.items() if f'goal_{colour}' in obj), None)
        goalPos, = goalPosDict.values()
        return self.calculateDistance(objPos[:2], goalPos[:2])
    
    def getDistRobToGoal(self, objID):
        objName, _ = next(((obj, pos[objID]) for (obj, pos) in self.positions.items() if objID in self.positions[obj]), None)
        colour = objName.split('_')[1]
        _, goalPos = next(((obj, pos[1]) for (obj, pos) in self.positions.items() if f'goal_{colour}' in self.positions), None)
        return self.calculateDistance(self.positions['robot'], goalPos[:2])

    def calcReward(self):
        '''returns rewards based on new distances'''
        self.positions = self.handleEnv.getPositions()
        self.prevDistRobToObj = self.distRobToObj
        self.prevNearObjectID = self.nearObjectID

        self.distRobToObj, self.nearObjectID = self.getNearestObjectToRobot()
        if self.nearObjectID != self.prevNearObjectID: # new nearest Object
            self.prevDistRobToObj = self.distRobToObj
            self.prevDistObjToGoal = self.getDistObjToGoal(self.nearObjectID)
            self.prevDistRobToGoal = self.getDistRobToGoal(self.nearObjectID)
            print(f'new nearest obj with id {self.nearObjectID}!')
            return 15
        else: # still in touch with the object or on path to nearest object
            self.distObjToGoal = self.getDistObjToGoal(self.nearObjectID)
            self.distRobToGoal = self.getDistRobToGoal(self.nearObjectID)
            print(f'Rob to Object: {self.distRobToObj}')
            print(f'Obj to Goal: {self.distObjToGoal}')
            print(f'Rob to Goal: {self.distRobToGoal}')
            reward = 0
            if self.prevDistRobToObj - self.distRobToObj > 0.005:
                reward += 1 # reward getting closer to nearest object
            else:
                reward -= 1 # penalty opposite of above
            if self.prevDistObjToGoal - self.distObjToGoal > 0.005:
                reward += 1 # reward pushing object to goal
            else:
                reward -= 1 # penalty opposite of above
            if self.prevDistRobToGoal - self.distRobToGoal > 0.005:
                reward += 0.1 # reward robot getting closer to goal
            else:
                reward -= 0.1 # penalty opposite of above
            return reward
        
    
            
def main():
    hEnv = HandleEnvironment(render=True, assets_folder="/home/group1/workspace/assets")
    hEnv.spawnGoals()
    hEnv.spawnObjects()

    hEnv.robotToStartPose()
    calcRew = CalcReward(hEnv)
    reward = calcRew.calcReward()
    reward = calcRew.calcReward()
    reward = calcRew.calcReward()
    reward = calcRew.calcReward()
    reward = calcRew.calcReward()
    # ids = hEnv.getIDs()
    # pprint(ids)
  
    # # input('Press Enter to continue')
    # states = hEnv.getStates()
    # print('States:')
    # pprint(states)

    # input('Press Enter to continue')

if __name__ == '__main__':
    main()