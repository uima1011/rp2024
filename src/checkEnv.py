from sortingViaPushingEnv import sortingViaPushingEnv as svpEnv

env = svpEnv()

# Test 1:
# uncomment and comment rest of file for first check, then proceed with test 2
# check_env(env)

# Test 2:
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

episodes = 50
for episode in range(episodes):
    done = False
    obs = env.reset()
    while True: #not done:
        # action = env.action_space.sample() # random actions
        action = moveRobotKeyboard() # manual actions
        if action is not None:
            obs, reward, done, truncated, info = env.step(action)