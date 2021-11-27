# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters

m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        self.action_space = list(permutations([i for i in range(m)], 2))
        self.state_space = [[i, j, k] for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)
        self.action_size = m*(m-1) + 1 
        self.total_time = 0
        self.max_time = 24*30
        self.poisons = [2,12,4,7,8]
      


    ## Encoding state (or state-action) for NN input
    def state_trans(self, state):

        state_encod = np.zeros((m+t+d))
        state_encod[state[0]] = 1
        state_encod[m + np.int(state[1])] = 1
        state_encod[m + t + np.int(state[2])] = 2

        
        return state_encod



    def requests(self, state):
        """Determining the number of requests basis the location. Use the table specified in the problem and complete for rest of the locations"""
        location = state[0]
        requests = self.poisons[location]    
        if requests >15:
            requests = 15

        possible_actions_index = random.sample(range(self.action_size-1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]

       # [0, 0] is not a 'request', but it is one of the possible actions
        actions.append([0,0])
        possible_actions_index.append(0)
        return possible_actions_index, actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start_loc, time, day = state
        pickup, drop = action


        if pickup == 0 and drop == 0:
            return -5
        else:
            """calculate the reward for the (pickup, drop) kind of actions"""
            time_elapsed_till_pickup = Time_matrix[start_loc, pickup, int(time), int(day)]

            # when pickup is not same as current location, current time and day could change

            time_next = (time +  time_elapsed_till_pickup) % t
            day_next = (day + (time +  time_elapsed_till_pickup)//t) % d

            return (R*Time_matrix[pickup, drop, int(time_next),int(day_next)] - C*(Time_matrix[pickup, drop, int(time_next), int(day_next)] + Time_matrix[start_loc, pickup, int(time), int(day)]))




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        start_loc, time, day = state
        pickup, drop = action
        
        
        if pickup == 0 and drop == 0:
            # when action is (0,0)
            time_elapsed = 1
            
            self.total_time = self.total_time + time_elapsed
        else:

            # when pickup is not same as current location, current time and day could change

            time_elapsed_till_pickup = Time_matrix[start_loc, pickup, int(time), int(day)]
            time_next_temp = (time +  time_elapsed_till_pickup) % t
            day_next_temp = (day + (time +  time_elapsed_till_pickup)//t) % d

            time = time_next_temp
            day = day_next_temp

            time_elapsed = Time_matrix[pickup, drop,  int(time), int(day)]
            
            self.total_time = self.total_time +  time_elapsed + time_elapsed_till_pickup

        time_next = (time + time_elapsed)%t
        day_next = (day + (time + time_elapsed)//t) % d
        
        time_next = np.int(time_next)
        day_next = np.int(day_next)
        
        # check whether it is a terminal state
        if (self.total_time >= self.max_time):
            terminal_state = 1
            self.total_time = 0
        else:
            terminal_state =0

        terminal_state = bool(terminal_state) # returns terminal state as True or False

        next_state = [drop, time_next, day_next]
        
        return next_state, terminal_state
    

    def reset(self):
        return random.choice(self.state_space)