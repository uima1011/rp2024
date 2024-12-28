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
URDF_PATH = "/home/jovyan/workspace/assets/urdf/robot_without_gripper.urdf"

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
        self.action_space = gym.spaces.Discrete(4)  # 4 Bewegungen
        self.object_ids = []
        self.object_ids = []
        self.target_colors = []
        self.target_positions = {}
        self.state_dim = None  # Definieren, sobald Objekte erstellt werden
        self.observation_space = None  # Dynamisch gesetzt nach `spawn_objects()`

    
    def spawn_objects(self):
        self.object_ids = []  # Initialisiere die Liste der Objekt-IDs
        num_red_cube = np.random.randint(1, 4)
        num_green_cube = np.random.randint(1, 4)
        num_red_plus = np.random.randint(1, 4)
        num_green_plus = np.random.randint(1, 4)

        for i in range(num_red_cube):
            red_cube_urdf_path = "/home/jovyan/workspace/assets/objects/cube_red.urdf"
            x_r = np.random.uniform(0.3, 0.9)
            y_r = np.random.uniform(-0.29, 0.29)
            z_r = 0.1
            red_cube_pose = Affine(translation=[x_r, y_r, z_r])
            obj_id = bullet_client.loadURDF(
                red_cube_urdf_path,
                red_cube_pose.translation,
                red_cube_pose.quat,
                flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            self.object_ids.append(obj_id)

        for i in range(num_green_cube):
            green_cube_urdf_path = "/home/jovyan/workspace/assets/objects/cube_green.urdf"
            x_g = np.random.uniform(0.3, 0.9)
            y_g = np.random.uniform(-0.29, 0.29)
            z_g = 0.1
            green_cube_pose = Affine(translation=[x_g, y_g, z_g])
            obj_id = bullet_client.loadURDF(
                green_cube_urdf_path,
                green_cube_pose.translation,
                green_cube_pose.quat,
                flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            self.object_ids.append(obj_id)

        for i in range(num_red_plus):
            red_plus_urdf_path = "/home/jovyan/workspace/assets/objects/plus_red.urdf"
            x_r = np.random.uniform(0.3, 0.9)
            y_r = np.random.uniform(-0.29, 0.29)
            z_r = 0.1
            red_plus_pose = Affine(translation=[x_r, y_r, z_r])
            obj_id = bullet_client.loadURDF(
                red_plus_urdf_path,
                red_plus_pose.translation,
                red_plus_pose.quat,
                flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            self.object_ids.append(obj_id)

        for i in range(num_green_plus):
            green_plus_urdf_path = "/home/jovyan/workspace/assets/objects/plus_green.urdf"
            x_g = np.random.uniform(0.3, 0.9)
            y_g = np.random.uniform(-0.29, 0.29)
            z_g = 0.1
            green_plus_pose = Affine(translation=[x_g, y_g, z_g])
            obj_id = bullet_client.loadURDF(
                green_plus_urdf_path,
                green_plus_pose.translation,
                green_plus_pose.quat,
                flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            self.object_ids.append(obj_id)
            
        # Hier Zielbereiche und andere IDs speichern
        #self.target_colors = ["red"] * len(red_cube_id) + ["green"] * len(green_cube_id)  # Beispiel
        self.target_positions = {"red": [0.5, 0.5, 0.1], "green": [0.8, -0.5, 0.1]}  # Beispiel
        self.state_dim = 3 * len(self.object_ids) + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))

        return self.object_ids


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
        self.spawn_objects()
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


