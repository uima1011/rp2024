import time
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from bullet_env.bullet_robot import BulletRobot
from transform import Affine

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym

# setup
RENDER = True
URDF_PATH = "/home/jovyan/workspace/assets/urdf/robot_without_gripper.urdf"

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

bullet_client.resetSimulation()

robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)
robot.home()

def spawn_objects():
    
    num_red_cube = np.random.randint(1,4)
    num_green_cube = np.random.randint(1,4)
    num_red_plus = np.random.randint(1,4)
    num_green_plus = np.random.randint(1,4)
    

    red_cube_id = [0]*num_red_cube
    green_cube_id = [0]*num_green_cube
    red_plus_id = [0]*num_red_plus
    green_plus_id = [0]*num_green_plus


    for i in range(num_red_cube):
        # spawn a red object in area (0.3to0.9,0.29to-0.29,0.1)
        red_cube_urdf_path = "/home/jovyan/workspace/assets/objects/cube_red.urdf"
        # generate random coordinates within the specified area
        x_r = np.random.uniform(0.3, 0.9)
        y_r = np.random.uniform(-0.29, 0.29)
        z_r = 0.1

        # create the object pose using the random coordinates
        red_cube_pose = Affine(translation=[x_r, y_r, z_r])

        # load the red object URDF with the generated pose
        red_cube_id[i] = bullet_client.loadURDF(
            red_cube_urdf_path,
            red_cube_pose.translation,
            red_cube_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    for i in range(num_green_cube):
        # spawn a green object in area (0.3to0.9,0.29to-0.29,0.1)
        green_cube_urdf_path = "/home/jovyan/workspace/assets/objects/cube_green.urdf"
        # generate random coordinates within the specified area
        x_g = np.random.uniform(0.3, 0.9)
        y_g = np.random.uniform(-0.29, 0.29)
        z_g = 0.1

        # create the object pose using the random coordinates
        green_cube_pose = Affine(translation=[x_g, y_g, z_g])

        # load the green object URDF with the generated pose
        green_cube_id[i] = bullet_client.loadURDF(
            green_cube_urdf_path,
            green_cube_pose.translation,
            green_cube_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
    for i in range(num_red_plus):
        # spawn a red object in area (0.3to0.9,0.29to-0.29,0.1)
        red_plus_urdf_path = "/home/jovyan/workspace/assets/objects/plus_red.urdf"
        # generate random coordinates within the specified area
        x_r = np.random.uniform(0.3, 0.9)
        y_r = np.random.uniform(-0.29, 0.29)
        z_r = 0.1

        # create the object pose using the random coordinates
        red_plus_pose = Affine(translation=[x_r, y_r, z_r])

        # load the red object URDF with the generated pose
        red_plus_id[i] = bullet_client.loadURDF(
            red_plus_urdf_path,
            red_plus_pose.translation,
            red_plus_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
    for i in range(num_green_plus):
        # spawn a green object in area (0.3to0.9,0.29to-0.29,0.1)
        green_plus_urdf_path = "/home/jovyan/workspace/assets/objects/plus_green.urdf"
        # generate random coordinates within the specified area
        x_g = np.random.uniform(0.3, 0.9)
        y_g = np.random.uniform(-0.29, 0.29)
        z_g = 0.1

        # create the object pose using the random coordinates
        green_plus_pose = Affine(translation=[x_g, y_g, z_g])

        # load the green object URDF with the generated pose
        green_plus_id[i] = bullet_client.loadURDF(
            green_plus_urdf_path,
            green_plus_pose.translation,
            green_plus_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
    # simulate the scene for 100 steps and wait for the object to settle
    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)
    
    return [red_cube_id, green_cube_id, red_plus_id, green_plus_id]


# Functions to move the robot

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

def get_state(object_ids):
    # Sammle die Positionen der Objekte
    object_positions = [bullet_client.getBasePositionAndOrientation(obj_id)[0] for obj_id in object_ids]
    # Sammle die Position des Roboters
    robot_pose = robot.get_eef_pose().translation
    # Kombiniere diese Informationen
    state = np.concatenate([robot_pose, np.array(object_positions).flatten()])
    return state

def perform_action(action):
    if action == 0:  # Nach links schieben
        move_left()
    elif action == 1:  # Nach rechts schieben
        move_right()
    elif action == 2:  # Nach vorne schieben
        move_forward()
    elif action == 3:  # Nach hinten schieben
        move_backward()

def all_objects_sorted(object_ids, target_colors, target_positions):
    # Prüfe, ob alle Objekte sortiert sind
    return all([bullet_client.getBasePositionAndOrientation(obj_id)[0] == target_positions[target_color] for obj_id, target_color in zip(object_ids, target_colors)])

def compute_reward(object_ids, target_colors, target_positions):
    reward = 0
    for obj_id, target_color in zip(object_ids, target_colors):
        obj_pos = bullet_client.getBasePositionAndOrientation(obj_id)[0]
        target_pos = target_positions[target_color]
        distance = np.linalg.norm(np.array(obj_pos) - np.array(target_pos))
        reward -= distance  # Strafe für Entfernung
    if all_objects_sorted(object_ids, target_colors, target_positions):  # Prüfe, ob alle Objekte sortiert sind
        reward += 100  # Große Belohnung für das Lösen der Aufgabe
    return reward


# Environment erstellen
class PushingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)  # 4 Bewegungen
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
    
    def reset(self):
        bullet_client.resetSimulation()
        spawn_objects()
        return get_state()
    
    def step(self, action):
        perform_action(action)
        reward = compute_reward()
        done = all_objects_sorted()  # Aufgabe abgeschlossen
        return get_state(), reward, done, {}

# Training
''' 
env = DummyVecEnv([lambda: PushingEnv()])

# Training mit PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Speichern und Testen
model.save("pushing_policy")
'''

def main():

    ids = spawn_objects()

    # implement pushing the object
    # keep in mind, that the object pose is defined in the world frame, and the eef points downwards

    start_pose()
    print("Pose: ", robot.get_eef_pose())

    # wait for key press
    input("Press Enter to continue...")

    # close the simulation
    bullet_client.disconnect()

if __name__ == "__main__":
    main()