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
        # Passe dies entsprechend der maximal erwarteten Zustandsdimension an
        self.max_objects = 16  # Annahme: maximal 10 Objekte
        state_dim = 2 + 3 * self.max_objects + 3 * 2  # robot_state + object_states + goal_states
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.episode = 0
        self.step_count = 0
        self.log_path = "/home/group1/workspace/src/log.json"
        self.log_data = []
            
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
                
                # Print goal areas
                print(f"Goal area 1: {goal1_coords}")
                print(f"Goal area 2: {goal2_coords}")
                
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
        print("Robot pose (z koordinate):", robot_pose.translation[2])
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

    def calculate_reward(self):
        current_pose = robot.get_eef_pose()
        current_position = current_pose.translation

        # Get positions of all objects
        objects_positions = [bullet_client.getBasePositionAndOrientation(obj_id)[0] for obj_id_list in self.object_ids.values() for obj_id in obj_id_list]
        
        # Calculate distances to all objects
        distances = [np.linalg.norm(np.array(current_position) - np.array(obj_pos)) for obj_pos in objects_positions]
        
        # Find the nearest object
        nearest_distance = min(distances)
        nearest_object_id = self.object_ids[next(obj for obj in self.object_ids if nearest_distance in [np.linalg.norm(np.array(current_position) - np.array(bullet_client.getBasePositionAndOrientation(obj_id)[0])) for obj_id in self.object_ids[obj]])][0]
        print("Nearest object ID:", nearest_object_id)
        nearest_object_type = next(obj for obj in self.object_ids if nearest_distance in [np.linalg.norm(np.array(current_position) - np.array(bullet_client.getBasePositionAndOrientation(obj_id)[0])) for obj_id in self.object_ids[obj]])
        print("Nearest object type:", nearest_object_type)
        
        # Calculate reward
        if self.previous_distance is None or nearest_distance < self.previous_distance:
            reward = 1
        else:
            reward = 0
        
        self.previous_distance = nearest_distance
        return reward

    def compute_reward(self):  # TODO check if function is working / fix
        reward = self.calculate_reward()
        reward = 0
        for obj_id, target_color in zip(self.object_ids, self.target_colors):
            obj_pos = bullet_client.getBasePositionAndOrientation(obj_id)[0]
            target_pos = self.target_positions[target_color]
            distance = np.linalg.norm(np.array(obj_pos) - np.array(target_pos))
            reward -= distance
        if False: # TODO if no nearest object do...
            reward += 100
        # implement more rewards
        return reward
    
    def start_pose(self):
        robot.home()
        target_pose = Affine(translation=[0.25, -0.25, -0.1], rotation=[-np.pi, 0, np.pi/2])
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
        reward = np.random.uniform(-1000, 1000) #self.compute_reward() # TODO: for testing, uncomment reward function later
        
        max_steps = 1000 # TODO: think about max steps after which episode is terminated
        if self.step_count >= max_steps:
            truncated = True
        else:
            truncated = False
        info = {} # additional info
        # if(getNearestObjectRobot()==None):
        #   done = True
        # if no neuarest object = None, task is done
        done = False # TODO: terminal condition if goal is achieved

        print(f"\rEpisode {self.episode}, Step {self.step_count}: Reward: {reward}, Done: {done}, Action: {action}", end="")
        state = self.get_state()
        self.log_step(action, reward, state, done)
        return state, reward, done, truncated, info
    
def train(environment):
    # Umgebung erstellen
    env = DummyVecEnv([lambda: environment])

    # PPO-Modell initialisieren
    model = PPO("MlpPolicy", env, verbose=1)

    # Training starten
    # falls ein existierendes model weitertrainiert werden soll:
    # model = PPO.load("existing_pushing_policy", env=env)
    print("Training beginnt...")
    model.learn(total_timesteps=100)  # Anzahl der Trainingsschritte

    # Modell speichern
    model.save("pushing_policy")
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
    input("Press Enter to continue...")
    bullet_client.disconnect()

if __name__ == "__main__":
    main()
