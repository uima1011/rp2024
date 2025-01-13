from pprint  import pprint
from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
from stable_baselines3.common.env_checker import check_env
import sys



test = 2 # 1 = testEnv, 2 = controll robot with keyboard, 3 = random Actions in a loop

env = svpEnv()


def moveRobotKeyboard():
    manualTermination = False
    translatedInput = -1
    wasdInput = input("Move Robot with wasd")
    if wasdInput == 'w': # forwards
        translatedInput = 1
    elif wasdInput == 's': # backwards
        translatedInput = 0
    elif wasdInput == 'a': # left
        translatedInput = 2 
    elif wasdInput == 'd':
        translatedInput = 3
    elif wasdInput == 't':
        manualTermination = True
    else: # typing error
        return None, False
    return translatedInput, manualTermination

def main():
    if test == 3:
        _ = env.reset()
        while(True):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            pprint(info)
            if terminated or truncated:
                _ = env.reset()
    elif test == 2:
        episodes = 50
        for episode in range(episodes):
            terminated = False
            truncated = False
            obs = env.reset()
            if episode >0: 
                print(f"Error: Environment was resetted (presumably after misbehaviour of robot / objects). Episode: {episode}")
            try:
                while terminated is False and truncated is False: #not terminated:
                    # action = env.action_space.sample() # random actions
                    # state_obj_z = handle_objects_instance.get_state_obj_z()
                    # print(f"State object z: {state_obj_z}")
                    # input("Press Enter to continue...")
                    action, manTermination = moveRobotKeyboard() # manual actions
                    if manTermination:
                        break
                    if action is not None:
                        obs, reward, terminated, truncated, info = env.step(action)
            except KeyboardInterrupt:
                print("keyboard interrupt")
                env.close()
    else:
        check_env(env) # test 2
        env.close()

            
        

if __name__ == '__main__':
    main()