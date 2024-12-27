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

def spawn_objects(num_red, num_green):

    red_object_id = [0]*num_red
    green_object_id = [0]*num_green

    for i in range(num_red):
        # spawn a red object in area (0.3to0.9,0.29to-0.29,0.1)
        red_object_urdf_path = "/home/jovyan/workspace/assets/objects/cube/object_red.urdf"
        # generate random coordinates within the specified area
        x_r = np.random.uniform(0.3, 0.9)
        y_r = np.random.uniform(-0.29, 0.29)
        z_r = 0.1

        # create the object pose using the random coordinates
        red_object_pose = Affine(translation=[x_r, y_r, z_r])

        # load the red object URDF with the generated pose
        red_object_id[i] = bullet_client.loadURDF(
            red_object_urdf_path,
            red_object_pose.translation,
            red_object_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

    for i in range(num_green):
        # spawn a green object in area (0.3to0.9,0.29to-0.29,0.1)
        green_object_urdf_path = "/home/jovyan/workspace/assets/objects/cube/object_green.urdf"
        # generate random coordinates within the specified area
        x_g = np.random.uniform(0.3, 0.9)
        y_g = np.random.uniform(-0.29, 0.29)
        z_g = 0.1

        # create the object pose using the random coordinates
        green_object_pose = Affine(translation=[x_g, y_g, z_g])

        # load the green object URDF with the generated pose
        green_object_id[i] = bullet_client.loadURDF(
            green_object_urdf_path,
            green_object_pose.translation,
            green_object_pose.quat,
            flags=bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
    # simulate the scene for 100 steps and wait for the object to settle
    for _ in range(100):
        bullet_client.stepSimulation()
        time.sleep(1 / 100)
    
    return [red_object_id,green_object_id]



def main():

    ids = spawn_objects(2, 2)

    

    # implement pushing the object
    # keep in mind, that the object pose is defined in the world frame, and the eef points downwards

    # home pos: [0.69137079 0.1741478  0.47682923] [ 7.07106601e-01  7.07106959e-01 -4.75657114e-05 -3.24074868e-05]
    # start pos: [0.29137079 0.5741478  0.07682923] [ 7.07106601e-01  7.07106959e-01 -4.75657114e-05 -3.24074868e-05]


    current_pose = robot.get_eef_pose()
    print("Pose: ", current_pose)

    target_pose = current_pose * Affine(translation=[0, 0, 0.6])
    robot.lin(target_pose)
    time.sleep(1)
    target_pose = target_pose * Affine(translation=[0, -0.5, 0])
    robot.lin(target_pose)
    time.sleep(1)
    target_pose = target_pose * Affine(translation=[0.4, 0, 0])
    robot.lin(target_pose)


    # wait for key press
    input("Press Enter to continue...")

    # close the simulation
    bullet_client.disconnect()

if __name__ == "__main__":
    main()