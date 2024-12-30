import matplotlib.pyplot as plt
import json


class plotLogs():
    def __init__(self, max_object, log_path):
        self.maxObjects = max_object
        self.logPath = log_path
        self.logData = []
        self.rewards = []
        self.actions = []
        self.steps = []
        self.statesDict = {}

    def readJSON(self):
        with open(self.logPath, mode='r') as file:
            self.logData = json.load(file)

    def getEntrys(self):
        self.rewards = [entry["Reward"] for entry in self.logData]
        self.action = [entry["Action"] for entry in self.logData]
        self.steps = [entry["Step"] for entry in self.logData]
        self.states = [entry["State"] for entry in self.logData]

    def getStates(self):
        # Robot
        robot_states = [state[0:2] for state in self.states]
        robotStates_x = [state[0] for state in robot_states]
        robotStates_y = [state[1] for state in robot_states]
        # Objects
        object_states = [state[2:(2 + 3 * self.maxObjects)] for state in self.states]
        objectStates_x = []
        objectStates_y = []
        objectStates_deg = []
        for i in range(len(object_states[0])):
            if i % 3 == 0:  # every 3rd is x
                objectStates_x.append([state[i] for state in object_states])
            elif i % 3 == 1: # every 3rd is y
                objectStates_y.append([state[i] for state in object_states])
            elif i % 3 == 2: # every 3rd is angle z
                objectStates_deg.append([state[i] for state in object_states])
        # Goals
        goal_states = [state[(2 + 3 * self.maxObjects)::] for state in self.states]
        goal_states_x = []
        goal_states_y = []
        goal_states_deg = []
        for i in range(len(goal_states[0])):
            if i % 3 == 0:  # every 3rd is x
                goal_states_x.append([state[i] for state in goal_states])
            elif i % 3 == 1: # every 3rd is y
                goal_states_y.append([state[i] for state in goal_states])
            elif i % 3 == 2: # every 3rd is angle z
                goal_states_deg.append([state[i] for state in goal_states])
        states = {
            'Robot': {
                'x': robotStates_x,
                'y': robotStates_y
            },
            'Objects': {
                'x': objectStates_x,
                'y': objectStates_y,
                'deg': objectStates_deg
            },
            'Goals': {
                'x': goal_states_x,
                'y': goal_states_y,
                'deg': goal_states_deg
            }
        }
        self.statesDict =  states

    def plotRewardAction(self):
        fig, axs = plt.subplots(2, 1)
        fig.canvas.manager.set_window_title('Reward and Actions')
        axs[0].set_title("Rewards")
        axs[0].plot(self.rewards)
        axs[1].set_title("Actions")
        axs[1].plot(self.action)

    def plotRobotStates(self):
        states = self.statesDict['Robot']
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 15))
        fig.canvas.manager.set_window_title('Robot States')

        axs[0].set_title("x-coords")
        axs[0].plot([state for state in states['x']], label=f'x')

        axs[1].set_title("y-coords")
        axs[1].plot([state for state in states['y']], label=f'y')

    def plotObjectsStates(self):
        states = self.statesDict['Objects']
        x_axis = [i for i in range(len(states['x'][0]))]
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.canvas.manager.set_window_title('Object States')

        axs[0].set_title("x-coords")
        for i in range(self.maxObjects):
            axs[0].plot(x_axis, states['x'][i], label=f'x_{i}')
        axs[0].legend()

        axs[1].set_title("y-coords")
        for i in range(self.maxObjects):
            axs[1].plot(x_axis, states['y'][i], label=f'y_{i}')
        axs[1].legend()
                
        axs[2].set_title("angle")
        for i in range(self.maxObjects):
            axs[2].plot(x_axis, states['deg'][i], label=f'deg_{i}')
        axs[2].legend()

    def plotGoalsStates(self):
        states = self.statesDict['Goals']

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig.canvas.manager.set_window_title('Goal States')

        axs[0].set_title("x-coords")
        for i in range(len(states[0])):
            axs[0].plot([state for state in states['x']], label=f'x_{i}')
        axs[0].legend()

        axs[1].set_title("y-coords")
        for i in range(len(states[0])):
            axs[1].plot([state for state in states['y']], label=f'y_{i}')
        axs[1].legend()
                
        axs[2].set_title("angle")
        for i in range(len(states[0])):
            axs[2].plot([state for state in states['deg']], label=f'deg_{i}')
        axs[2].legend()

    def plotStates(self):
        self.plotRobotStates()
        self.plotObjectsStates()
        # self.plotGoalsStates()
        

    def plotLog(self):
        self.readJSON()
        self.getEntrys()
        self.getStates()
        self.plotRewardAction()
        self.plotStates()
        plt.show()

def main():
    max_object = 16
    log_path = r'/home/group1/workspace/src/log copy.json'
    pL = plotLogs(max_object, log_path)
    pL.plotLog()

if __name__ == '__main__':
    main()
    

