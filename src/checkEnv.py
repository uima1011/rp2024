from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv
from stable_baselines3.common.env_checker import check_env
env = svpEnv()

test = 1 # change to TEST 2 or 1

def moveRobotKeyboard():
    wasdInput = input("Move Robot with wasd")
    if wasdInput == 'w': # forwards
        translatedInput = 1
    elif wasdInput == 's': # backwards
        translatedInput = 0
    elif wasdInput == 'a': # left
        translatedInput = 2 
    elif wasdInput == 'd':
        translatedInput = 3
    else: # typing error
        return None
    return translatedInput

def main():
    if test == 1:
        episodes = 50
        for episode in range(episodes):
            done = False
            obs = env.reset()
            if episode >0: 
                print(f"Error: Environment was resetted (presumably after misbehaviour of robot / objects). Episode: {episode}")
            while done==False: #not done:
                # action = env.action_space.sample() # random actions
                # state_obj_z = handle_objects_instance.get_state_obj_z()
                # print(f"State object z: {state_obj_z}")
                # input("Press Enter to continue...")
                action = moveRobotKeyboard() # manual actions
                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
    else:
        check_env(env) # test 2

if __name__ == '__main__':
    main()