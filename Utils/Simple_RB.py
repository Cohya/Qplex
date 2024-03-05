import numpy as np 

class EpisodeRpelayBuffer():
    def __init__(self, capacity):
        self.transitions = [] 
        self.next_idx = 0
        self.replay_buffer = [None] * capacity 
        self.capacity = capacity 
        self.currect_capacity = 0
        self.rng = np.random.default_rng()
      
    def store_transition(self, state, action, reward, next_state, done, time, full_state_info):
        transition = [state, action, reward, next_state, done, time, full_state_info]
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
        
        b_state = []
        b_actions = []
        b_rewards = []
        next_state = []
        b_dones = []
        b_times = []
        b_full_state_info = []
        
        #Generate random  idx for smapling in a uniformed manner from all games
        idxs = self.rng.integers(low = 0 , high = self.currect_capacity, size = batch_size)
        
        #Statrt gattering the data for the training 
        for idx in idxs:
            choosen_game_transitions = self.replay_buffer[idx]    
            for transition in  choosen_game_transitions:
                states, actions, reward, next_states, done, time, full_state_info = transition
                
                b_state.append(states)
                b_actions.append(actions)
                b_rewards.append(reward)
                next_state.append(next_states)
                b_dones.append(done)
                b_times.append(time)
                b_full_state_info.append(full_state_info)
                
        return b_state, np.array(b_actions),\
               np.array(b_rewards), next_state, np.array(b_dones),\
                   np.array(b_times), np.array(b_full_state_info)
                
    