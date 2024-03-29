from QplexNet import QplexNet
import numpy as np 
import tensorflow as tf 
import yaml
from Utils.dotdict import dotdict
from pettingzoo.mpe import simple_adversary_v3
import time
from Games.simple_matrix import SimpleMatrix
from Simple_RB import EpisodeRpelayBuffer
from AgentClass import Agent


def convert_to_full_state(obs):
    for i, s0 in enumerate(obs):
        s_i = tf.one_hot(s0, obervation_dims)
        s_i = tf.expand_dims(tf.expand_dims(s_i, axis = 0), axis = 0)
                
        if i == 0 :
                full_state = s_i
        else:
            full_state = tf.concat((full_state, s_i), axis = -1)
            
    return full_state.numpy()
            
tf.config.set_visible_devices([],'GPU')
with open("configconfigSimpleGame.yaml", "r") as file:
    args= yaml.safe_load(file)
    
args = dotdict(args)


obs = tf.random.normal(shape = (2,1,1))# 3 agents each one with observation (1,5)
states = tf.reshape(obs, shape = (1,2*1)) #  the state is obs * 3 flatten 
h =  tf.zeros(shape = (2, 1, 64)) # just the hidden for GRU cell (num_of_agents, dims)
actions = [0,1]# Actions that they took 
number_of_agents = args.number_of_agents # 2 
agents = []
for i in range(number_of_agents):
    agents.append(Agent(args))
    
qp_main = QplexNet(agents = agents,args = args,  share_weights = True) # this is with share 
qp_target =  QplexNet(agents = agents,args = args, share_weights = True)


q_tot = qp_main.get_q_tot(obs, hidden_state = h, states = states, actions = actions)

rb = EpisodeRpelayBuffer(capacity = 1000)

env = SimpleMatrix()
obs_n = env.observation_sapce
action_n = env.actions_space 
obervation_dims = 1

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
for episode in range(1000):
    obs = env.reset()
    
    actions = [None, None]
    h_vec = [None, None]
    observations = [None, None]
    next_obs_to_save = [None, None]
    done = False
    while not done:
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i], eps = 1.0)
            
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
        
        
        
        with tf.GradientTape(watch_accessed_variables= True) as tape:
                
            b_observations_updated = []
            b_next_obs_updated = []
            for i in range(batch_size):
                obs =  b_observations[i]
                b_observations_updated.append(np.array(obs))
                
                next_obs = b_next_obs[i]
                b_next_obs_updated.append(np.array(next_obs))
                
            
            hidden_state_vec  = []
            hidden_state = None
            for i in  range(batch_size):
                
                q_tot, h = qp_main.get_q_tot(b_observations_updated[i],
                                  hidden_state = hidden_state,
                                  states = b_full_state[i][0,...],
                                  actions = b_actions[i])
                
                # print(q_tot)
            
                # Here I should remember the hidden state ! 
                if i == 0 :
                    
                    q_tot_vec = q_tot
                    
                else:
                    q_tot_vec = tf.concat((q_tot_vec, q_tot), axis = 0)
                
                
               
                if b_dones[i] == True:
                    hidden_state = None
                    # print("done[i]:", b_dones[i])
                else:
                    hidden_state = h
                
                
                
            ### Calculate the target 
            targets = []
            gamma = 0.99
            for j in  range(batch_size):
             
                _, h = qp_target.get_q_tot(b_observations_updated[i],
                                  hidden_state = hidden_state,
                                  states = b_full_state[i][0,...],
                                  actions = b_actions[i])
                
                done = b_dones[i] 
                if done == True:
                    hidden_state = None
                    # print("done[i]:", b_dones[i])
                else:
                    hidden_state = h
                
                
                adv_i, v_i  = qp_target.q_target(b_next_obs_updated[i],
                                  hidden_state = hidden_state,
                                  states = b_full_state[i][0,...])
                
                actions_i = list(np.argmax(adv_i, axis = 1))
            
                q_tot,h = qp_target.get_q_tot(b_observations_updated[i],
                                  hidden_state = hidden_state,
                                  states = b_full_state[i][0,...],
                                  actions = actions_i)
                
                
                
                target = b_rewards[i] + gamma * q_tot * np.invert(done).astype(np.float32)
                targets.append(target[0].numpy())
            
            targets = tf.stop_gradient(np.array(targets))
            loss = tf.reduce_mean(tf.math.square(q_tot_vec - targets))
            
        gradients = tape.gradient(loss, qp_main.trainable_params)
        
        optimizer.apply_gradients(zip(gradients,qp_main.trainable_params))
        
        
        ## check the correctness 
        obs  = (0,0) 
        observations = [None, None]
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            # agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i], eps = 0.5)
            

            h_vec[i] =  h 
        actions = [0,0]
        full_state = convert_to_full_state(obs)
        observations = np.array(observations)
        q_tot_test,_ = qp_main.get_q_tot(observations,
                          hidden_state = None,
                          states = full_state[0],
                          actions = actions)
        
        print("episode:", episode)
        print("loss:", loss.numpy())
        print("q_tot[0,0]:", q_tot_test.numpy()[0])
        
        obs  = (0,0) 
        observations = [None, None]
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            # agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i], eps = 0.5)
            
            h_vec[i] =  h 
        actions = [1,1]
        full_state = convert_to_full_state(obs)
        observations = np.array(observations)
        q_tot_test,_ = qp_main.get_q_tot(observations,
                          hidden_state = None,
                          states = full_state[0],
                          actions = actions)
        print("q_tot[1,1]:", q_tot_test.numpy()[0])
        
        
        obs  = (0,0) 
        observations = [None, None]
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            # agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i], eps = 0.5)
            
            h_vec[i] =  h 
            
        actions = [0,1]
        full_state = convert_to_full_state(obs)
        observations = np.array(observations)
        q_tot_test,_ = qp_main.get_q_tot(observations,
                          hidden_state = None,
                          states = full_state[0],
                          actions = actions)
        print("q_tot[1,0]:", q_tot_test.numpy()[0])
        
        
        obs  = (0,0) 
        observations = [None, None]
        for i, s0 in enumerate(obs):
            s_i = tf.one_hot(s0, obervation_dims)
            s_i = np.expand_dims(s_i.numpy(), axis = 0)
            # s_i = np.expand_dims(, axis = 0)
            
            observations[i] = s_i
            
            # agent_action, h = agents[i].sample_action(s_i,  hidden_state = h_vec[i], eps = 0.5)
            
            h_vec[i] =  h 
            
        actions = [2,1]
        full_state = convert_to_full_state(obs)
        observations = np.array(observations)
        q_tot_test,_ = qp_main.get_q_tot(observations,
                          hidden_state = None,
                          states = full_state[0],
                          actions = actions)
        print("q_tot[2,1]:", q_tot_test.numpy()[0])
        
    if episode % 10 == 0:
        ## copy params - hard copy 
        for w_main, w_target in zip(qp_main.trainable_params, qp_target.trainable_params):
            w_target.assign(w_main.numpy())