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





def main():

    ids = spawn_objects()

    

    # implement pushing the object
    # keep in mind, that the object pose is defined in the world frame, and the eef points downwards

    # home pos: [0.69137079 0.1741478  0.47682923] [ 7.07106601e-01  7.07106959e-01 -4.75657114e-05 -3.24074868e-05]
    
    start_pose()

    move_forward()
    move_left()
    move_forward()
    move_left()
    move_forward()
    move_left()
    move_forward()
    move_left()
    move_forward()
    move_left()
    move_forward()
    move_left()
    move_forward()
    move_left()

    current_pose = robot.get_eef_pose()
    print("Pose: ", current_pose)


    # wait for key press
    input("Press Enter to continue...")

    # close the simulation
    bullet_client.disconnect()

if __name__ == "__main__":
    main()