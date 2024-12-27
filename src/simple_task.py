import time
import pybullet as p
import numpy as np
from pybullet_utils.bullet_client import BulletClient

from bullet_env.bullet_robot import BulletRobot, BulletGripper
from transform import Affine

# setup
RENDER = True
URDF_PATH = "/home/jovyan/workspace/assets/urdf/robot.urdf"

bullet_client = BulletClient(connection_mode=p.GUI)
bullet_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
if not RENDER:
    bullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

bullet_client.resetSimulation()

robot = BulletRobot(bullet_client=bullet_client, urdf_path=URDF_PATH)
gripper = BulletGripper(bullet_client=bullet_client, robot_id=robot.robot_id)
robot.home()

# spawn an object
object_urdf_path = "/home/jovyan/workspace/assets/objects/cube/object.urdf"
object_pose = Affine(translation=[0.5, 0, 0.1])
object_id = bullet_client.loadURDF(
    object_urdf_path,
    object_pose.translation,
    object_pose.quat,
    flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

# simulate the scene for 100 steps and wait for the object to settle
for _ in range(100):
    bullet_client.stepSimulation()
    time.sleep(1 / 100)

# implement grasping the object
# keep in mind, that the object pose is defined in the world frame, and the eef points downwards
# also, make sure that before grasping the gripper is open
# consider adding a pre-grasp pose to ensure the object is grasped correctly without collision during approach

# wait for key press
input("Press Enter to continue...")

# close the simulation
bullet_client.disconnect()
