from Qplex  import Qplex
import numpy as np 
import tensorflow as tf 
import yaml
from Utils.dotdict import dotdict
from pettingzoo.mpe import simple_adversary_v3
import time
import tensorflow as tf 

tf.config.set_visible_devices([],'GPU')

with open("config.yaml", "r") as file:
    args= yaml.safe_load(file)
    
args = dotdict(args)


obs = tf.random.normal(shape = (3,1,5))# 3 agents each one with observation (1,5)
states = tf.reshape(obs, shape = (1,3*5)) #  the state is obs * 3 flatten 
h =  tf.zeros(shape = (3, 1, 64)) # just the hidden for GRU cell
actions = [1,2,3]# Actions that they took 

qp_main = Qplex(args)
qp_target =  Qplex(args)


q_tot = qp_main.get_q_tot(obs, hidden_state = h, states = states, actions = actions)



env = simple_adversary_v3.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {}
    for agent in  env.agents:
        if agent == 'adversary_0':
            actions[agent] = 0

    observations, rewards, terminations, truncations, infos = env.step(actions)
    time.sleep(1)
env.close()
