import gym
import numpy as np
import random
env = gym.make('CartPole-v1')
env.reset()

#There are two states, the observations are pos or neg
#There are two actions in the states, left or right 
state_list = [[0,1],[0,1]]
states = np.array(state_list)
e_rate = 0.1

#Initalize q_values for each action and state to 0
q_values = np.zeros((2,2))

best_steps = 0

#Number of episodes to complete
for i_episode in range(100):
        observation = env.reset() #Reset environment
        
        #Sum the values to get an idea of inital state
        values = sum(observation)
        #Set state based on the values
        if values >= 0:
            curr_state = 1
        else:
            curr_state = 0

        #Number of timesteps per episode
        for t in range(100):
            env.render()

            #Do a greedy action more % than the e_rate
            if (random.random() > e_rate):
                #Action is determined by the max q value
                #Since only actions are 0 or 1, just using index for action
                action = max((value, index) for index, value in\
                             enumerate(q_values[curr_state]))[1]
            #Otherwise, do a random action
            else:
                action = states[curr_state][random.randint(0,1)]
            #Perform the action
            observation, reward, done, info = env.step(action)

            #Sum the values to get an idea of the state
            values = sum(observation)
            #Set state based on the values
            if values >= 0:
                curr_state = 1
            else:
                curr_state = 0

            #Now update the q values
            q_values[curr_state][action] = reward + max(value for value in \
                                            q_values[curr_state])

            if done:
                if best_steps < t:
                    best_steps = t
                break
print(best_steps)
