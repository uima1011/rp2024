import time
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# Setup
RENDER = True
URDF_PATH = "/home/group1/workspace/assets/urdf/robot_without_gripper.urdf"

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

bullet_client.resetSimulation()
robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)
robot.home()


# Environment
class PushingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tableCords = {
                            'x':[0.3, 0.9], # min, max
                            'y':[-0.29, 0.29]
                          }
        self.goal_width = {
                            'x': 0.05 + 0.01, # min_size_object + offset
                            'y': 0.05 + 0.01
                          }     
        self.action_space = gym.spaces.Discrete(4)  # 4 Bewegungen

        self.object_ids = []
        self.target_colors = []
        self.target_positions = {}
        self.state_dim = None  # Definieren, sobald Objekte erstellt werden
        self.observation_space = None  # Dynamisch gesetzt nach `spawn_objects()`

    def check_rectangle_overlap(self, rect1, rect2):
        """
        Check if two rectangles overlap.
        Each rectangle is defined by (x_min, y_min, x_max, y_max)
        """
        # If one rectangle is on the left side of the other
        if rect1[2] < rect2[0] or rect2[2] < rect1[0]:
            return False
        # If one rectangle is above the other
        if rect1[3] < rect2[1] or rect2[3] < rect1[1]:
            return False
        return True

    def generate_single_goal_area(self, table_coords, goal_width):
        """Generate coordinates for a single goal area"""
        x_goal_min = np.random.uniform(table_coords['x'][0], table_coords['x'][1] - goal_width['x'])
        y_goal_min = np.random.uniform(table_coords['y'][0], table_coords['y'][1] - goal_width['y'])
        x_goal_max = x_goal_min + goal_width['x']
        y_goal_max = y_goal_min + goal_width['y']
        
        return (x_goal_min, y_goal_min, x_goal_max, y_goal_max)

    def generate_goal_pose(self, goal_coords, z_goal):
        """Generate pose for a goal area"""
        x_goal_min, y_goal_min, x_goal_max, y_goal_max = goal_coords
        mid_x = (x_goal_max - x_goal_min) / 2 + x_goal_min
        mid_y = (y_goal_max - y_goal_min) / 2 + y_goal_min
        
        translation = [mid_x, mid_y, z_goal]
        rotation = [0, 0, np.random.uniform(0, 2*np.pi)]  # around z axis
        return Affine(translation, rotation)

    def generateGoalAreas(self, colours=['red', 'green'], max_attempts=100):
        """
        Generate two non-overlapping goal areas.
        Returns False if unable to generate non-overlapping areas after max_attempts.
        """
        z_goal = -0.01
        
        for attempt in range(max_attempts):
            # Generate first goal area
            goal1_coords = self.generate_single_goal_area(self.tableCords, self.goal_width)
            # Generate second goal area
            goal2_coords = self.generate_single_goal_area(self.tableCords, self.goal_width)
            
            # Check if they overlap
            if not self.check_rectangle_overlap(goal1_coords, goal2_coords):
                # Generate poses for both goals
                goal1_pose = self.generate_goal_pose(goal1_coords, z_goal)
                goal2_pose = self.generate_goal_pose(goal2_coords, z_goal)
                
                # Print goal areas
                print(f"Goal area 1: {goal1_coords}")
                print(f"Goal area 2: {goal2_coords}")
                
                # Load URDFs
                for coords, pose, colour in zip([goal1_coords, goal2_coords], 
                                            [goal1_pose, goal2_pose], 
                                            colours):
                    urdf_path = f"/home/group1/workspace/assets/objects/goals/goal_{colour}.urdf"
                    bullet_client.loadURDF(
                        urdf_path,
                        pose.translation,
                        pose.quat,
                        flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                    )
                return True
                
        print("Failed to generate non-overlapping goal areas after", max_attempts, "attempts")
        return False


    def check_collision(self, pos1, other_positions, min_distance=0.1):
        """
        Check if a position is too close to any existing positions
        Args:
            pos1: [x, y, z] position to check
            other_positions: list of existing [[x, y, z], ...] positions
            min_distance: minimum allowed distance between objects
        Returns:
            bool: True if collision detected, False otherwise
        """
        for pos2 in other_positions:
            distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1[:2], pos2[:2])))  # Only check x,y
            if distance < min_distance:
                return True
        return False

    def generate_valid_position(self, existing_positions, min_distance=0.1, max_attempts=100):
        """Generate a random position that doesn't collide with existing objects"""
        for _ in range(max_attempts):
            x = np.random.uniform(0.3, 0.9)
            y = np.random.uniform(-0.29, 0.29)
            z = 0.1
            
            if not self.check_collision([x, y, z], existing_positions, min_distance):
                return [x, y, z]
        return None

    def spawn_single_object(self, urdf_path, bullet_client, existing_positions):
        """Spawn a single object at a random position avoiding collisions"""
        position = self.generate_valid_position(existing_positions)
        if position == None:
            return None
        existing_positions.append(position) #add to list of positions
        
        pose = Affine(translation = position)
        
        return bullet_client.loadURDF(
            urdf_path,
            pose.translation,
            pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )

    def spawn_objects(self, bullet_client, folders, objects, colours, maxObjCount):
        """Spawn random numbers of cubes and plus signs in different colors with collision checking"""
        existing_positions = []
        # Define object configurations
        base_path = "/home/group1/workspace/assets/objects/"
        object_ids = {'cube_red': [], 'cube_green': [], 'plus_red': [], 'plus_green': []} # init with empty lists
        for folder, obj in zip(folders, objects):
            for col in colours:
                objName = obj + '_' + col
                path = base_path + folder + f'/{objName}.urdf'
                objCount = np.random.randint(1, maxObjCount)
                for _ in range(objCount):
                    objID = self.spawn_single_object(path, bullet_client, existing_positions)
                    if objID is not None:
                        object_ids[objName].append(objID)
                    else:
                        print(f"Warning: Could not place {objName} - skipping remaining objects of this type")
                        break 
        
        # Let objects settle
        for _ in range(100):
            bullet_client.stepSimulation()
            time.sleep(1/100)
        
        # TODO implement correct:
        # Hier Zielbereiche und andere IDs speichern
        #self.target_colors = ["red"] * len(red_cube_id) + ["green"] * len(green_cube_id)  # Beispiel
        self.target_positions = {"red": [0.5, 0.5, 0.1], "green": [0.8, -0.5, 0.1]}  # Beispiel
        self.state_dim = 3 * len(self.object_ids) + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))

        # Return IDs in the same format as original function
        return self.object_ids # TODO check if order rly is same

    def get_state(self):
        object_positions = [
            bullet_client.getBasePositionAndOrientation(obj_id)[0]
            for obj_id in self.object_ids
        ]
        robot_pose = robot.get_eef_pose()
        robot_pose = robot_pose.translation
        return np.concatenate([robot_pose, np.array(object_positions).flatten()])

    def perform_action(self, action):
        if action == 0:
            move_left()
        elif action == 1:
            move_right()
        elif action == 2:
            move_forward()
        elif action == 3:
            move_backward()

    def all_objects_sorted(self):
        return all(
            [
                bullet_client.getBasePositionAndOrientation(obj_id)[0] == self.target_positions[target_color]
                for obj_id, target_color in zip(self.object_ids, self.target_colors)
            ]
        )

    def compute_reward(self):
        reward = 0
        for obj_id, target_color in zip(self.object_ids, self.target_colors):
            obj_pos = bullet_client.getBasePositionAndOrientation(obj_id)[0]
            target_pos = self.target_positions[target_color]
            distance = np.linalg.norm(np.array(obj_pos) - np.array(target_pos))
            reward -= distance
        if self.all_objects_sorted():
            reward += 100
        return reward

    def reset(self):
        bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)  # Roboter neu laden
        robot.home()
        maxObjCount = 4
        self.spawn_objects(bullet_client, ['cubes', 'signs'], ['cube', 'plus'], ['red', 'green'], maxObjCount)
        return self.get_state()

    def step(self, action):
        self.perform_action(action)
        reward = self.compute_reward()
        done = self.all_objects_sorted()
        return self.get_state(), reward, done, {}

# Funktionen für Bewegungen
def move_right():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[-0.1, 0, 0])
    robot.lin(target_pose)

def move_left():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[0.1, 0, 0])
    robot.lin(target_pose)

def move_forward():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[0, 0.1, 0])
    robot.lin(target_pose)

def move_backward():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[0, -0.1, 0])
    robot.lin(target_pose)

def move_up():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[0, 0, -0.1])
    robot.lin(target_pose)

def move_down():
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[0, 0, 0.1])
    robot.lin(target_pose)

def start_pose():
    robot.home()
    current_pose = robot.get_eef_pose()
    target_pose = current_pose * Affine(translation=[-0.55, -0.45, 0.6])
    robot.lin(target_pose)

def main():
    env = PushingEnv()
    env.reset()
    input("Press Enter to continue...")
    bullet_client.disconnect()

if __name__ == "__main__":
    main()



# Struktur für späteres Training (in etwa)
"""
env = DummyVecEnv([lambda: PushingEnv()])

# Training mit PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Speichern und Testen
model.save("pushing_policy")
"""


