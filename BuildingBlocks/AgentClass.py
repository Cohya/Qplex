import tensorflow as tf 
import ast 
import numpy as np 
from collections import deque

class Agent(tf.keras.Model):
    
    def __init__(self,args):
        super().__init__()
        self.action_dims = args.action_dims
        self.rnn_hidden_dims = args.rnn_hidden_dims
        self.fc1 = tf.keras.layers.Dense(args.rnn_embedded_dims,
                                         activation=tf.nn.relu,
                                         use_bias = True)
        self.rnn = tf.keras.layers.GRUCell(self.rnn_hidden_dims)
        self.fc2 = tf.keras.layers.Dense(args.action_dims, activation = None,
                                         use_bias = True)
        
        self.history = 10
        
        ## activate the networks
        input_shape = ast.literal_eval(args.rnn_input_dims)
      
        x = tf.random.normal(shape = input_shape)
        q,h=  self.__call__(x)
        
    def __call__(self, obs, hidden_state = None, training = False):
        if hidden_state == None:
            hidden_state = tf.zeros(shape = (1, self.rnn_hidden_dims))
            # print("hidden_state:", hidden_state.shape)
            
        x = self.fc1(obs)
        # hidden_state = tf.reshape(hidden_state, (-1, self.rnn_hidden_dims))
        # print("hidden_state:", hidden_state.shape)
        # print("x:", x.shape)
        h, _ = self.rnn(x, hidden_state)
        q = self.fc2(h)
        q = q[:,0,:]
        return q, h 
    
    def sample_action(self, obs, hidden_state, eps = 0, training = False):
        
        ## check the size  of the input 
        size_dims = len(np.shape(obs))
        if  size_dims == 2:
            obs = np.expand_dims(obs, axis = 0)
        
        elif size_dims > 3 or size_dims < 2:
            assert False, "observation  dimention  is wrong !!"
        q, h = self.__call__(obs,  hidden_state, training)
        
        
        if np.random.random()<eps:
            action  = np.random.choice(self.action_dims)
        else:
            action  =  np.argmax(q[0])
            
        return action, h
    
    
    def reset(self, obs):
        self.trajectory = self._init_deque(obs)
        self.obs = obs
        self.prev_obs = obs 
        
    def _init_deque(self, obs):
        trajectory = deque(maxlen = self.history)
        for _ in range(self.history):
            trajectory.append(obs)
        return trajectory
     
    def load_weights(self,w):
        for w_i, w_local in zip(w, self.trainable_variables):
            w_local.assign(w_i)
    
    def get_weights(self):
        weights = []
        
        for w in self.trainable_variables:
            weights.append(w.numpy())
        
        return weights
        