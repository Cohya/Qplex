

class SimpleMatrix:
    def __init__(self,):
        print('activte simple game')
        
        self.realQval = [[8,-12,-12],
                         [-12, 6, 6],
                         [-12, 0,  6]]
        
        self.observation_sapce = 0
        self.actions_space = [0,1,2]
        
    def step(self,actions):
        
        if actions[0] == 0 and actions[1] == 0:
            r =  8
        elif (actions[0] == 0 and actions[1] == 1) or (actions[0] == 1 and actions[1] == 0) :
            r = -12
        
        elif (actions[0] == 0 and actions[1] == 2) or (actions[0] == 2 and actions[1] == 0) :
            r =  -12  
        
        elif (actions[0] == 1 and actions[1] == 1):
            r= 6
        
        elif (actions[0] == 2 and actions[1] == 2):
            r= 6
        
        elif (actions[0] == 2 and actions[1] == 1) or (actions[0] == 1 and actions[1] == 2):
            r= 0
        
        else:
            print("Error")
            assert False,  "actions are not valide"
        
        next_state = (0,0)
        done = True
        info = {}
        return next_state,  r, done ,info
    
    
    def reset(self):
        self.state = (0,0)
        return self.state