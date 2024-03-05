import sys 
from QplexBasedTF.QplexNet import QplexNet
import tensorflow as tf 
import numpy as np 




class Qplex():
    def __init__(self, agents, args, share_weights = True):
        self.qplex_main = QplexNet(agents = agents, args = args, share_weights = share_weights)
        self.qplex_target = QplexNet(agents = agents, args = args, share_weights = share_weights)
        self.gamma = args.gamma 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
        
        
    def calc_loss(self,batch_size, b_full_state, b_observations_updated, b_actions, b_rewards, 
                       b_full_next_state, b_next_obs_updated,
                       b_dones):
                    
        hidden_state = None
        for i in  range(batch_size):
            
            q_tot, h = self.qplex_main.get_q_tot(b_observations_updated[i],
                              hidden_state = hidden_state,
                              states = b_full_state[i][0,...],
                              actions = b_actions[i])
            
        
            # Here I should remember the hidden state ! 
            if i == 0 :
                
                q_tot_vec = q_tot
                
            else:
                q_tot_vec = tf.concat((q_tot_vec, q_tot), axis = 0)
            
            
           
            if b_dones[i] == True:
                hidden_state = None
            else:
                hidden_state = h
                
            
        ### Calculate the target 
        targets = []
        
        for i in  range(batch_size):
            
            _, h = self.qplex_target.get_q_tot(b_observations_updated[i],
                              hidden_state = hidden_state,
                              states = b_full_state[i][0,...],
                              actions = b_actions[i])
            
            done = b_dones[i] 
            if done == True:
                hidden_state = None
                # print("done[i]:", b_dones[i])
            else:
                hidden_state = h
            
            
            adv_i, v_i  = self.qplex_target.q_target(b_next_obs_updated[i],
                              hidden_state = hidden_state,
                              states = b_full_state[i][0,...])
            
            actions_i = list(np.argmax(adv_i, axis = 1))
        
            q_tot,h = self.qplex_target.get_q_tot(b_observations_updated[i],
                              hidden_state = hidden_state,
                              states = b_full_state[i][0,...],
                              actions = actions_i)
            
            
         
            target = b_rewards[i] + self.gamma * q_tot * np.invert(done).astype(np.float32)
            targets.append(target[0].numpy())
        
        targets = tf.stop_gradient(np.array(targets))
        
        loss = tf.reduce_mean(tf.math.square(q_tot_vec - targets))
        
        return loss 
    
    def update_weights(self, batch_size, b_full_state, b_observations_updated, b_actions, b_rewards, 
                       b_full_next_state, b_next_obs_updated,
                       b_dones):
        
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            
            loss = self.calc_loss(batch_size,
                                  b_full_state,
                                  b_observations_updated,
                                  b_actions,
                                  b_rewards, 
                                  b_full_next_state,
                                  b_next_obs_updated, 
                                  b_dones)
            
        gradients = tape.gradient(loss, self.qplex_main.trainable_params)
        
        self.optimizer.apply_gradients(zip(gradients,self.qplex_main.trainable_params))
        
        return loss
    def target_hard_update(self):
        for w_main,  w_target in zip(self.qplex_main.trainable_params, 
                                     self.qplex_target.trainable_params):
            
            w_target.assign(w_main)
            
        