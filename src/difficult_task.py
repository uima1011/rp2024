import time
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient
from bullet_env.bullet_robot import BulletRobot
from transform import Affine
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
import json
import os

# Setup
RENDER = True
URDF_PATH = "/home/group1/workspace/assets/urdf/robot_without_gripper.urdf"

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

bullet_client.resetSimulation()
robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)


# Environment
class PushingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tableCords = {
                            'x':[0.3, 0.9], # min, max
                            'y':[-0.29, 0.29]
                          }
        self.goal_width = {
                            'x': 0.15 + 0.01, # min_size_object + offset
                            'y': 0.15 + 0.01
                          }     
        self.action_space = gym.spaces.Discrete(4)  # 4 directions (up, down, left, right)
        self.object_ids = {}
        self.goal_ids = {}
        self.target_positions = {}
        # Beispielhafte Platzhalter-Definition von observation_space
        self.max_objects = 16  # Annahme: maximal 10 Objekte
        state_dim = 2 + 3 * self.max_objects + 3 * 2  # robot_state + object_states + goal_states
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.episode = 0
        self.step_count = 0
        self.log_path = "/home/group1/workspace/src/log.json"
        self.log_data = []
        self.distance = [None, None, None]
        self.previous_distance = [None, None, None]
        self.nearest_object_id = None
        self.previous_nearest_object_id = None
            
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
        obj_id = {'goal_red': [], 'goal_green': []}
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
                
                # Load URDFs
                for coords, pose, colour in zip([goal1_coords, goal2_coords], 
                                            [goal1_pose, goal2_pose], 
                                            colours):
                    urdf_path = f"/home/group1/workspace/assets/objects/goals/goal_{colour}.urdf"
                    objID = bullet_client.loadURDF(
                        urdf_path,
                        pose.translation,
                        pose.quat,
                        flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                    )
                    obj_id[f'goal_{colour}'].append(objID)
                self.goal_ids = obj_id
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
        angle = np.random.uniform(0, 2*np.pi)
        pose = Affine(translation = position, rotation=(0, 0, angle))
        
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
        self.object_ids = object_ids 
        
        # Let objects settle
        for _ in range(100):
            bullet_client.stepSimulation()
            time.sleep(1/100)
        
        # Hier Zielbereiche und andere IDs speichern
        num_objects = sum(len(obj_id_list) for obj_id_list in self.object_ids.values())
        self.state_dim = (2 + 2 * num_objects + num_objects + 2 * 2 + 2,) # robot_positions(x,y) + object_positions(x,y)*num_objects + object_orientations + goal_positions + goal_oriantations

        # simulate the scene for 100 steps and wait for the object to settle
        for _ in range(100):
            bullet_client.stepSimulation()
            time.sleep(1 / 100)
    
    def get_state(self):
        object_states = []
        for obj_id_list in self.object_ids.values():
            for obj_id in obj_id_list:
                position, orientation = bullet_client.getBasePositionAndOrientation(obj_id)
                # Extrahiere den Winkel um die Z-Achse aus der Quaternion
                euler_angles = R.from_quat(orientation).as_euler('xyz')
                angle_z = euler_angles[2]  # Winkel um die Z-Achse
                object_states.append([position[0], position[1], angle_z])

        goal_states = []
        for goal_id_list in self.goal_ids.values():
            for goal_id in goal_id_list:
                position, orientation = bullet_client.getBasePositionAndOrientation(goal_id)
                # Extrahiere den Winkel um die Z-Achse aus der Quaternion
                euler_angles = R.from_quat(orientation).as_euler('xyz')
                angle_z = euler_angles[2]
                goal_states.append([position[0], position[1], angle_z])

        # Extrahiere die Roboterpose
        robot_pose = robot.get_eef_pose()
        robot_position = robot_pose.translation[:2]  # Nur x, y

        robot_state = np.array([robot_position[0], robot_position[1]])

        # Fülle Zustände auf die maximale Dimension auf
        object_states = np.array(object_states).flatten()
        object_states = np.pad(object_states, (0, 3 * self.max_objects - len(object_states)), constant_values=0)

        goal_states = np.array(goal_states).flatten()
        # Zustand zusammensetzen
        return np.concatenate([robot_state, object_states, goal_states])
    
    def perform_action(self, action):
        current_pose = robot.get_eef_pose()
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
        robot.lin(target_pose)
    
    def get_dist_robot_object(self, obj_id):
        current_pose = robot.get_eef_pose()
        current_position = current_pose.translation
        obj_position = bullet_client.getBasePositionAndOrientation(obj_id)[0]
        distance = np.linalg.norm(np.array(current_position) - np.array(obj_position))
        return distance
    
    def get_dist_robot_all_objects(self):
        dist_list = []
        for obj_id_list in self.object_ids.values():
            for obj_id in obj_id_list:
                self.get_dist_robot_object(obj_id)
                dist_list.append(self.get_dist_robot_object(obj_id))
        return dist_list

    def get_dist_object_goal(self, obj_id):
        nearest_object_type = next((obj for obj in self.object_ids if obj_id in self.object_ids[obj]), None) # get key which contains obj id
        if nearest_object_type is None: # id not found in dict
            raise ValueError(f"Object ID {obj_id} not found in object_ids.")
        
        nearest_object_color = nearest_object_type.split('_')[1] # get colour from key name
        goal_id = self.goal_ids[f'goal_{nearest_object_color}'][0] # get goal id by colour
        goal_position = bullet_client.getBasePositionAndOrientation(goal_id)[0] # get goal position by id
        obj_position = bullet_client.getBasePositionAndOrientation(obj_id)[0] # get object position by id

        return np.linalg.norm(np.array(obj_position) - np.array(goal_position)) # calc distance betweeen object and goal

    def object_inside_goal(self, obj_id): 
        # Check if object is inside goal area
        dist_def_inside_goal = 0.05
        if self.get_dist_object_goal(obj_id) < dist_def_inside_goal:
            return True
        else:
            return False
    
    def get_dist_robot_goal(self, nearest_obj_id):
        current_pose = robot.get_eef_pose()
        current_position = current_pose.translation
        if nearest_obj_id is None:
            return None
        nearest_object_type = next(obj for obj in self.object_ids if nearest_obj_id in self.object_ids[obj])
        nearest_object_color = nearest_object_type.split('_')[1]
        goal_id = self.goal_ids[f'goal_{nearest_object_color}'][0]
        goal_position = bullet_client.getBasePositionAndOrientation(goal_id)[0]
        distance_robot_goal = np.linalg.norm(np.array(current_position) - np.array(goal_position))
        return distance_robot_goal

    def get_nearest_object_to_robot(self):
        while True:
            if self.nearest_object_id is None: # wenn kein nearest object bekannt (erster Schritt oder letztes im Ziel)
                dist_robot_all_objects = self.get_dist_robot_all_objects()
                # prüfe ob Array leer
                if not dist_robot_all_objects:
                    # TODO evlt self.done = True oder so
                    return None, None   # alle Objekte im Ziel --> done
                while True:
                    min_dist_robot_obj = min(dist_robot_all_objects)
                    nearest_object_index = dist_robot_all_objects.index(min_dist_robot_obj)
                    nearest_object_id = [obj_id for obj_id_list in self.object_ids.values() for obj_id in obj_id_list][nearest_object_index]
                    if not self.object_inside_goal(nearest_object_id):
                        return min_dist_robot_obj, nearest_object_id
                    dist_robot_all_objects.pop(nearest_object_index)
                    if not dist_robot_all_objects:  # prüfe erneut ob Array leer
                        return None, None
            else: # wenn neares object bekannt: Abstandsberechnung
                if self.object_inside_goal(self.nearest_object_id): # check if nearest object is inside goal area
                    self.nearest_object_id = None
                else: # Abstand berechnen
                    dist = self.get_dist_robot_object(self.nearest_object_id)
                    return dist, self.nearest_object_id    
                    return dist, self.nearest_object_id    

    def distances_for_reward(self):
        self.previous_nearest_object_id = self.nearest_object_id
        # dictance robot to nearest object 
        self.distance[0], self.nearest_object_id = self.get_nearest_object_to_robot()
        if (self.nearest_object_id != self.previous_nearest_object_id):
            # set previous distance to new nearest obj
            self.previous_distance[0] = self.distance[0]
            self.previous_distance[1] = self.get_dist_object_goal(self.nearest_object_id)
            self.previous_distance[2] = self.get_dist_robot_goal(self.nearest_object_id)
            reward = 15
            return reward
        # distance of that object to its goal
        self.distance[1] = self.get_dist_object_goal(self.nearest_object_id)
        #distance of robot to goal for nearest object
        self.distance[2] = self.get_dist_robot_goal(self.nearest_object_id)
        #remeber distances for next step
        reward = 0
        if abs(self.previous_distance[0] - self.distance[0]) > 0.005:
            reward += 0.9
        elif abs(self.distance[0] - self.previous_distance[0]) > 0.005:
            reward -= 0.9
        if abs(self.previous_distance[1] - self.distance[1]) > 0.005:
            reward += 5
        elif abs(self.distance[1] - self.previous_distance[1]) > 0.005:
            reward -= 0
        if abs(self.previous_distance[2] - self.distance[2]) > 0.005:
            reward -= 0.1
        elif abs(self.distance[2] - self.previous_distance[2]) > 0.005:
            reward += 0.1
        
        self.previous_distance = self.distance.copy()

        return reward
    
    def objectOffTable(self): 
        for obj_id_list in self.object_ids.values():
            for obj_id in obj_id_list:
                position, _ = bullet_client.getBasePositionAndOrientation(obj_id)
                if position[2] < 0:
                    return True
        # as soon as one object falls off the table, function gives True            
        return False
    
    def start_pose(self):
        robot.home()
        target_pose = Affine(translation=[0.25, -0.34, -0.1], rotation=[-np.pi, 0, np.pi/2])
        robot.lin(target_pose)

    def reset(self, seed = None):
        super().reset(seed=seed)
        bullet_client.resetSimulation()
        self.robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)  # Roboter neu laden
        self.start_pose()
        maxObjCount = 4
        self.spawn_objects(bullet_client, ['cubes', 'signs'], ['cube', 'plus'], ['red', 'green'], maxObjCount)
        self.generateGoalAreas()
        self.episode += 1
        self.step_count = 0
        self.distance = [None, None, None]
        self.previous_distance = [None, None, None]
        self.nearest_object_id = None
        self.previous_nearest_object_id = None

        # Berechne die Beobachtungsraumdimension basierend auf dem Zustand
        state = self.get_state()
        # Warnung, falls Zustand nicht mit observation_space übereinstimmt
        if state.shape != self.observation_space.shape:
            print(f"Warnung: Zustandsdimension ({state.shape}) stimmt nicht mit observation_space ({self.observation_space.shape}) überein.")

        info = {}  # Hier kannst du zusätzliche Informationen hinzufügen, falls benötigt
        return state, info

    def log_step(self, action, reward, state, done):
        log_entry = {
            "Episode": int(self.episode),
            "Step": int(self.step_count),
            "Action": int(action),
            "Reward": float(reward),
            "State": state.tolist() if isinstance(state, np.ndarray) else state,
            "Done": bool(done)
        }
        self.log_data.append(log_entry)
        with open(self.log_path, mode='w') as file:
            json.dump(self.log_data, file, indent=4)
        self.step_count += 1

    def step(self, action):
        self.perform_action(action)
        reward = self.distances_for_reward()

        max_steps = 99 # TODO: think about max steps after which episode is terminated
        if self.step_count >= max_steps:
            truncated = True
        else:
            truncated = False
        info = {} # additional info
        # if(getNearestObjectRobot()==None):
        #   done = True
        # if no neuarest object = None, task is done
        done = False # checks every step if at least one object has fallen off the table 
        # no matter which object has fallen off the table
        if(self.objectOffTable()):
            done = True
            reward = -1000

        print(f"\rEpisode {self.episode}, Step {self.step_count}: Reward: {reward}, Done: {done}, Action: {action}", end="")
        state = self.get_state()
        self.log_step(action, reward, state, done)
        return state, reward, done, truncated, info
    
def train(environment):
    TIMESTEPS = 200 # Anzahl der Trainingsschritte
    MODEL = "PPO"
    models_dir = f"/home/group1/workspace/data/models/{MODEL}"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    logdir = "/home/group1/workspace/data/logs"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # Umgebung erstellen
    env = environment #DummyVecEnv([lambda: environment])

    # PPO-Modell initialisieren
    model = PPO("MlpPolicy", env, gamma = 0.99, ent_coef=0.001, verbose=1, n_steps=100, tensorboard_log=logdir)
 
    # Training starten
    # falls ein existierendes model weitertrainiert werden soll:
    # model = PPO.load("/home/group1/workspace/data/train/pushing_policy_new_3", env=env, verbose=1, tensorboard_log=logdir)
    print("Training beginnt...")
    for i in range(1, 5): # limit to 30*TIMESTEPS = 30k steps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")  # Anzahl der Trainingsschrittes
        model.save(f'{models_dir}/{TIMESTEPS*i}')
        print(f'Iteration abgeschlossen: {i}')
    print("Training abgeschlossen und Modell gespeichert.")

    # Testphase (optional)
    '''
    test_env = PushingEnv()
    obs = test_env.reset()
    for _ in range(100):  # 100 Test-Schritte
        action, _ = model.predict(obs)
        obs, reward, done, _ = test_env.step(action)
        print("Reward:", reward)
        if done:
            print("Episode abgeschlossen")
            break
    '''      

def main():
    env = PushingEnv()
    train(env)

if __name__ == "__main__":
    main()