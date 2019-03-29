import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6,gamma=1,alpha=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.alpha = alpha
        
    def select_action(self, state,episode_i):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 1/episode_i
    
        probs = np.ones(self.nA)*(epsilon/self.nA)
        probs[np.argmax(self.Q[state])] = (1-epsilon)+(epsilon/self.nA)
    
        action = np.random.choice(np.arange(self.nA),p=probs)
    
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        if done:
            self.Q[state][action] = self.Q[state][action] + self.alpha*(reward-self.Q[state][action])
            return 
        
        d_reward = reward + self.gamma*(np.amax(self.Q[next_state]))
        self.Q[state][action] = self.Q[state][action] + self.alpha*(d_reward - self.Q[state][action])
        return 