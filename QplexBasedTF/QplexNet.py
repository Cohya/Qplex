
import tensorflow as tf 
# import argparse
# import yaml 
# from Utils.dotdict import dotdict
# import ast
# import numpy as np 
# tf.config.set_visible_devices([], 'GPU')

# from AgentClass import Agent 

class Transformation(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        
        ### For computing w
        self.w_fc1 = tf.keras.layers.Dense(args.hypernet_embed)
        self.w_fc2 = tf.keras.layers.Dense(args.number_of_agents)
        
        ### for computing b 
        self.b_fc1 = tf.keras.layers.Dense(args.hypernet_embed)
        self.b_fc2 = tf.keras.layers.Dense(args.number_of_agents)
        ## Initialized the network 
        s = tf.random.normal(shape = (1, args.state_dims))
        self.compute_w(s)
        self.compute_b(s)
      
    def compute_w(self, s):
        w = self.w_fc1(s)
        w = tf.nn.relu(w)
        w =  self.w_fc2(w)
        w = tf.math.abs(w) + 1e-10
        return w
    
    def compute_b(self, s):
        b = self.b_fc1(s)
        b = tf.nn.relu(b)
        b = self.b_fc2(b)
        return b 
 

class Generate_lambdas(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.number_of_heads = args.number_of_heads
        self.n_actions = args.action_dims
        self.state_dim = args.state_dims#int(tf.reduce_prod(ast.literal_eval(args.state_shape)))
        self.n_agents = args.number_of_agents
        # print("state_dim:", self.state_dim, self.n_actions, args.number_of_agents)
        self.action_dim = args.number_of_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        
        adv_hypernet_embed = args.adv_hypernet_embed
        
        self.num_kernel = args.number_of_heads
        
        self.key_extractors  = []
        self.agents_extractors = []
        self.action_extractors = []
        
        for _ in range(self.num_kernel):
            #state_dim as input
            self.key_extractors.append(tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(1),
                    ]
                    )
                )
            ##state_dim as input
            self.agents_extractors.append(tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(self.n_agents),
                    ]
                    )
                
                )
            # state_action_dim
            self.action_extractors.append(tf.keras.Sequential(
                    [
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(adv_hypernet_embed),
                        tf.keras.layers.ReLU(),
                        tf.keras.layers.Dense(self.n_agents),
                    ]
                    )
                )
        
        
        ## Activaet the layers 
        states = tf.random.normal(shape = (1, self.state_dim))
        actions =  tf.random.normal(shape  = (1,  self.action_dim))
        
        heads = self.call(states, actions)
        
        
    def call(self,states, actions):
        states =  tf.reshape(states, (-1, self.state_dim))
        actions = tf.reshape(actions, (-1, self.action_dim))
        data = tf.concat((states, actions), axis = 1)
        
        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [k_ext(states) for k_ext in self.agents_extractors]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]
        # print(all_head_action)
        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = tf.abs(curr_head_key)
            x_key = tf.tile(x_key, multiples=[1, self.n_agents]) + 1e-10
            # print(x_key.shape)
            x_agents = tf.nn.sigmoid(curr_head_agents)
            x_action = tf.nn.sigmoid(curr_head_action)
            # print(x_agents.shape)
            weights = x_key * x_agents * x_action ## <-- lambdas  >0
            head_attend_weights.append(weights)
        
       
        head_attend = tf.stack(head_attend_weights, axis = 1)
        # print("head_attend:", head_attend.shape)
        head_attend = tf.reshape(head_attend, shape = (-1, self.num_kernel, self.n_agents))
        # print("head_attend:", head_attend.shape)
        head_attend = tf.reduce_sum(head_attend, axis = 1)
       
        return head_attend
  
    
class QplexNet(tf.keras.Model):
    def __init__(self,agents,  args, share_weights = True ):
        #  Networks 
        super().__init__()
        self.agents = agents# AgentNet(args)
        # self.agent_networks = agents# AgentNet(args)
        self.duplex_dueling = Duplex_Dueling(args)
        
        self.trainable_params = []
        self.share_weights = share_weights
        if self.share_weights:
            self._share_weights_fnc()
            self.trainable_params += self.agents[0].trainable_variables
            
        else:
            for agent in self.agents:
                self.trainable_params += self.agents.trainable_variables
            
            
        self.trainable_params += self.duplex_dueling.trainable_variables
        
        
    def _share_weights_fnc(self):
        for i, agent in enumerate(self.agents):
            if i == 0:
                weights = agent.get_weights()
                
            else:
                agent.load_weights(weights)
                
    def q_target(self,  obs, hidden_state = None, states = None):
        
        if self.share_weights:
            adv_i, v_i,_  = self.get_Ai_and_Vi(self.agents[0], obs, hidden_state)
            
            
        return adv_i, v_i   
    
    
    def get_Ai_and_Vi(self, agent, obs, hidden_state = None): # tau -> touple of history and actions (O(t), a(t-1))
        q , h  = agent(obs, hidden_state)
        v_i = tf.expand_dims(tf.reduce_max(q,axis = -1), axis = 1)
        adv_i = q - v_i
        return adv_i, v_i, h
    
    def get_q_tot(self, obs, hidden_state = None, states = None, actions = None):
        
        if self.share_weights:
            adv_i, v_i, h_i = self.get_Ai_and_Vi(self.agents[0], obs, hidden_state)
            # print(adv_i.shape, v_i.shape)
            
        else:
            
            for i, agent in enumerate(self.agents):
                if i ==0:
                    print(obs[i,...])
                    adv_i , v_i, h_i = self.get_Ai_and_Vi(agent, 
                                                     obs[i:i+1,...], hidden_state[i:i+1,...])
                else:
                    adv_j , v_j, h_j = self.get_Ai_and_Vi(agent, obs[i:i+1,...], hidden_state[i:i+1,...])
                    adv_i = tf.concat((adv_i, adv_j), axis = 0 )
                    v_i = tf.concat((v_i, v_j), axis = 0 )
                    h_i = tf.concat((h_i,h_j), axis = 0)
                    
        q_tot =  self.duplex_dueling.get_q_tot(adv_i = adv_i,
                                               v_i = v_i,  
                                               states = states,
                                               actions = actions)   
        ## Check if it is working 
        return q_tot, h_i
        
        
class Duplex_Dueling(tf.keras.Model):
    def __init__(self,args):
        super().__init__()
        self.args = args
        #  Networks 
        self.generate_lambdas = Generate_lambdas(args)
        # self.agent_network = AgentNet(args)
        self.transformation_block = Transformation(args)
        
        # Trainable variables 
        self.trainable_params = [] 
        # self.trainable_params +=  self.agent_network.trainable_variables
        self.trainable_params += self.transformation_block.trainable_variables
        self.trainable_params += self.generate_lambdas.trainable_variables
        
        self.number_of_actions = args.action_dims
        
    # def get_Ai_and_Vi(self, obs, hidden_state = None): # tau -> touple of history and actions (O(t), a(t-1))
    #     q , h  = self.agent_network(obs, hidden_state)
    #     v_i = tf.expand_dims(tf.reduce_max(q,axis = -1), axis = 1)
    #     adv_i = q - v_i
    #     return adv_i, v_i  
        
    def transformation(self, adv_i, v_i, s_t, actions):
        # v_i = [number_of_agents, 1]
        # adv_i = [number_of_agent,number_of_actions]
        # actions = list of intigers  []
        w = self.transformation_block.compute_w(s_t)
        b = self.transformation_block.compute_b(s_t)
        v_i_tau_global = tf.transpose(w) * v_i + tf.transpose(b) # (number_of_agents, 1)
        adv_i_tau_global = tf.transpose(w) * adv_i # (numebr_of_agents,  num_of_actions)
        actions_one_hot = tf.one_hot(actions, depth = self.number_of_actions) # (number_of_agents, num_of_actions)
        temp = tf.math.multiply(adv_i_tau_global, actions_one_hot)
        adv_i_tau_global_for_actions =  tf.expand_dims(tf.reduce_sum(temp, axis = 1), axis = 1)#(number_of_agent,1)
        
        return  adv_i_tau_global_for_actions, v_i_tau_global 
        
    def dueling_mixing(self, adv_i_tau_global_for_actions, v_i_tau_global, s_t, actions):
        v_tot = tf.reduce_sum(v_i_tau_global)
        n, n_agents =  adv_i_tau_global_for_actions.shape 
        adv_i_tau_global_for_actions = tf.reshape(adv_i_tau_global_for_actions, shape = (n_agents, n))
        states = s_t
        lambdas_i_tau_global = self.generate_lambdas(states, actions)
        adv_tot = tf.reduce_sum(adv_i_tau_global_for_actions * (lambdas_i_tau_global - 1.), axis=1)
        # print(adv_tot.shape, lambdas_i_tau_global.shape, lambdas_i_tau_global.shape, adv_i_tau_global_for_actions.shape)
        q_tot =  adv_tot + v_tot
        return q_tot

    def get_q_tot(self, adv_i, v_i , states = None, actions = None):
        # adv_i, v_i = self.get_Ai_and_Vi(obs, hidden_state)
        
        # Transformation
        adv_i_tau_global_for_actions, v_i_tau_global = self.transformation(adv_i, 
                                                                           v_i, 
                                                                           states,
                                                                           actions)
        
        
        ## Dueling Mixing
        actions = tf.one_hot(indices = actions, depth = self.args.action_dims)
        q_tot = self.dueling_mixing(adv_i_tau_global_for_actions,
                                    v_i_tau_global, 
                                    states, 
                                    actions)
        
        return q_tot
        
        
# with open("config.yaml", "r") as file:
#     args= yaml.safe_load(file)
    
# args = dotdict(args)


# # net = AgentNet(args)
# # obs = tf.random.normal(shape = (1,1,5))
# # h0 =  tf.zeros(shape = (1, 1, 64))

# # q, h = net(obs, hidden_state=h0) # [3,10], 
# # v = tf.expand_dims(tf.reduce_max(q,axis = -1), axis = 1) # [3,1]

# # net_transformation = Transformation(args)
# # s =  tf.random.normal(shape =  (1,15))
# # b  = net_transformation.compute_b(s)
# # w =  net_transformation.compute_w(s)

# # v_i_tau_global = tf.transpose(w) * v + tf.transpose(b)


# # dd = Duplex_Dueling(args)

# # # adv_i, v_i = dd.get_Ai_and_Vi(obs,h)

# # v_i_tau_global = tf.transpose(w) * v_i + tf.transpose(b)
# # adv_i_tau_global = tf.transpose(w) * adv_i

# # actions = [1,2,3]
# # number_of_actions = 10
# # actions_one_hot = tf.one_hot(actions, depth = number_of_actions) 
# # temp = tf.math.multiply(adv_i_tau_global, actions_one_hot)
# # adv_i_tau_global_for_actions =  tf.reduce_sum(temp, axis = 1)

# obs = tf.random.normal(shape = (3,1,5))# 3 agents each one with observation (1,5)
# states = tf.reshape(obs, shape = (1,15)) #  the state is obs * 3 flatten 
# h =  tf.zeros(shape = (3, 1, 64)) # just the hidden for GRU cell
# actions = [1,2,3] # Actions that they took 
# agents = [Agent(args), Agent(args)]
# qp = QplexNet(agents,args)
# q_tot = qp.get_q_tot(obs, hidden_state = h, states = states, actions = actions)
