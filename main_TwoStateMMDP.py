# from QplexNet import QplexNet
from QplexBasedTF.Qplex  import Qplex
import numpy as np 
import tensorflow as tf 
import yaml
from Utils.dotdict import dotdict
from pettingzoo.mpe import simple_adversary_v3
import time
from Games.TwoStateMMDP import TwoStateMMDP
from BuildingBlocks.Simple_RB import EpisodeRpelayBuffer
from BuildingBlocks.AgentClass import Agent

"""
###################################
#            SimpleMatrix
##################################
Example Figure 3(a) in the original paper 
    "QPLEX: DUPLEX DUELING MULTI-AGENT Q-LEARNING""
    s1<-------s2
    
        
"""

def convert_to_full_state(obs):
    for i, s0 in enumerate(obs):
        s_i = tf.one_hot(s0, obervation_dims)
        s_i = tf.expand_dims(tf.expand_dims(s_i, axis = 0), axis = 0)
                
        if i == 0 :
                full_state = s_i
        else:
            full_state = tf.concat((full_state, s_i), axis = -1)
            
    return full_state.numpy()
            
# tf.config.set_visible_devices([],'GPU')
with open("configurations/TwoStateMMDP.yaml", "r") as file:
    args= yaml.safe_load(file)
    
args = dotdict(args)


# obs = tf.random.normal(shape = (2,1,1))# 3 agents each one with observation (1,5)
# states = tf.reshape(obs, shape = (1,2*1)) #  the state is obs * 3 flatten 
# h =  tf.zeros(shape = (2, 1, 64)) # just the hidden for GRU cell (num_of_agents, dims)
# actions = [0,1]# Actions that they took 
number_of_agents = args.number_of_agents # 2 
agents = []
for i in range(number_of_agents):
    agents.append(Agent(args))
    

qplex = Qplex(agents, args, share_weights = True)

rb = EpisodeRpelayBuffer(capacity = 5000)

env = TwoStateMMDP()
obs_n = env.observation_sapce
action_n = env.actions_space 
obervation_dims = 1

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
eps = 1.0
total_steps = 0
for episode in range(10000):
    obs = env.reset()
    
    actions = [None, None]
    h_vec = [None, None]
    observations = [None, None]
    next_obs_to_save = [None, None]
    done = False
    while not done:
        total_steps += 1
        eps = max(0.001, eps - 1.0/1000)
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i],
                                                      eps = eps)
            
            actions[i] = agent_action
            h_vec[i] =  h 
            
        
        full_state = convert_to_full_state(obs)
        next_obs, r, done, info= env.step(actions = actions)
        
        full_state_next = convert_to_full_state(next_obs)
         
        
        for i, s0 in enumerate(next_obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            next_obs_to_save[i] = s_i 
            
        rb.store_transition(full_state,
                        observations.copy(),
                        h_vec.copy(),
                        actions.copy(), 
                        r, 
                        full_state_next, 
                        next_obs_to_save.copy(), 
                        done,
                        info
                        )
        
        obs = next_obs
        
     
    if episode > 32:
        batch_size = 32        
        (b_full_state, b_observations, b_hidden, b_actions, b_rewards, 
               b_full_next_state, b_next_obs,
               b_dones, b_full_state_info) = rb.get_minibatch(batch_size= batch_size) 
        
        
        # b_full_state list of size = batch_size , shape = (1,1,2)
        # b_observations --> each element (1,1,1) [agent 1, agent 2]
        # b_hidden [agent 1, agent 2] --> [1,1,64]
        
        b_observations_updated = []
        b_next_obs_updated = []
        for i in range(batch_size):
            obs =  b_observations[i]
            b_observations_updated.append(np.array(obs))
            
            next_obs = b_next_obs[i]
            b_next_obs_updated.append(np.array(next_obs))
                
        loss = qplex.update_weights(batch_size,
                             b_full_state, 
                             b_observations_updated,
                             b_actions,
                             b_rewards, 
                             b_full_next_state,
                             b_next_obs_updated,
                             b_dones)
        
        if episode % 10 == 0:
            qplex.target_hard_update()
            
        ########################
        ## check the correctness 
        ########################
        obs  = (1,1) 
        observations = [None, None]
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            

            h_vec[i] =  h 
        actions = [0,0]
        full_state = convert_to_full_state(obs)
        observations = np.array(observations)
        q_tot_test,_ = qplex.qplex_main.get_q_tot(observations,
                          hidden_state = None,
                          states = full_state[0],
                          actions = actions)
        
        print("episode:", episode)
        print("loss:", loss.numpy())
        # print("q_tot[0,0]:", np.round(q_tot_test.numpy()[0], decimals =2),  " should be:", 8)
        print(f"q_tot[0,0]: {q_tot_test.numpy()[0]:.2f} should be: {100}")
        
       
        