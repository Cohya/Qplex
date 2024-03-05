import numpy as np 

class EpisodeRpelayBuffer():
    def __init__(self, capacity):
        self.transitions = [] 
        self.next_idx = 0
        self.replay_buffer = [None] * capacity 
        self.capacity = capacity 
        self.currect_capacity = 0
        self.rng = np.random.default_rng()
      
    def store_transition(self, full_state, 
                         observations, 
                         h_vec,
                         actions,
                         reward,
                         full_state_next, 
                         next_obs,
                         done,
                         full_state_info = None):
        
        transition = [full_state, 
                             observations, 
                             h_vec,
                             actions,
                             reward,
                             full_state_next, 
                             next_obs,
                             done,
                             full_state_info]
        
        self.transitions.append(transition)
        
        if done == True:
            # inser the whole game into the buffer 
            self.store_game()

    def store_game(self,):
        self.replay_buffer[self.next_idx] = list(self.transitions)
        self.next_idx = (self.next_idx +  1) % self.capacity 
        self.currect_capacity = min(self.currect_capacity + 1 , self.capacity)
        self.transitions = []
        
    def get_minibatch(self,batch_size = 64):
        assert  batch_size < self.currect_capacity, "Your replay buffer has lower games stored than in the required Batch size!"
        
        b_full_state = []
        b_observations = []
        b_hidden = []
        b_actions = []
        b_rewards = []
        b_full_next_state = []
        b_next_obs = []
        b_dones = []
        b_full_state_info = []
        
        #Generate random  idx for smapling in a uniformed manner from all games
        idxs = self.rng.integers(low = 0 , high = self.currect_capacity, size = batch_size)
        
        #Statrt gattering the data for the training 
        for idx in idxs:
            choosen_game_transitions = self.replay_buffer[idx]    
            for transition in  choosen_game_transitions:
                (full_state, 
                observations, 
                h_vec,
                actions,
                reward,
                full_state_next, 
                next_obs,
                done,
                full_state_info) = transition

                        
                b_full_state.append(full_state)
                b_observations.append(observations)
                b_hidden.append(h_vec)
                b_actions.append(actions)
                b_rewards.append(reward)
                b_full_next_state.append(full_state_next)
                b_next_obs.append(next_obs)
                b_dones.append(done)
                b_full_state_info.append(full_state_info)
        

        return  b_full_state, b_observations, \
                b_hidden, b_actions, b_rewards, \
                b_full_next_state, b_next_obs, \
                b_dones, b_full_state_info
    
    
    # b_state, np.array(b_actions),\
    #            np.array(b_rewards), next_state, np.array(b_dones),\
    #                np.array(b_times), np.array(b_full_state_info)
                
    