import time
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from bullet_env.bullet_robot import BulletRobot
from transform import Affine

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

# spawn a red object in area (0.3to0.9,0.29to-0.29,0.1)
red_object_urdf_path = "/home/jovyan/workspace/assets/objects/cube/object_red.urdf"
# generate random coordinates within the specified area
x_r = np.random.uniform(0.3, 0.9)
y_r = np.random.uniform(-0.29, 0.29)
z_r = 0.1

# create the object pose using the random coordinates
red_object_pose = Affine(translation=[x_r, y_r, z_r])

# load the red object URDF with the generated pose
red_object_id = bullet_client.loadURDF(
    red_object_urdf_path,
    red_object_pose.translation,
    red_object_pose.quat,
    flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

# spawn a green object in area (0.3to0.9,0.29to-0.29,0.1)
green_object_urdf_path = "/home/jovyan/workspace/assets/objects/cube/object_green.urdf"
# generate random coordinates within the specified area
x_g = np.random.uniform(0.3, 0.9)
y_g = np.random.uniform(-0.29, 0.29)
z_g = 0.1

# create the object pose using the random coordinates
green_object_pose = Affine(translation=[x_g, y_g, z_g])

# load the green object URDF with the generated pose
green_object_id = bullet_client.loadURDF(
    green_object_urdf_path,
    green_object_pose.translation,
    green_object_pose.quat,
    flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)


# simulate the scene for 100 steps and wait for the object to settle
for _ in range(100):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)

# implement pushing the object
# keep in mind, that the object pose is defined in the world frame, and the eef points downwards

#Beispliebewegung
current_pose = robot.get_eef_pose()
print("Pose: ", current_pose)

target_pose = current_pose * Affine(translation=[0, 0 ,0.1])
robot.lin(target_pose)

# wait for key press
input("Press Enter to continue...")

# close the simulation
bullet_client.disconnect()