
class TwoStateMMDP():
    def __init__(self):
        self.current_state = None
        self.observation_sapce = 0
        self.actions_space = [0,1]
        
        
    def step(self,actions):
        self.t += 1
        if self.t == 100:
            done = True
            
        else:
            done = False
            
            
        a1 = actions[0]
        a2 = actions[1]
        
        if self.current_state == 0: # 0 is s1 in the original game 
            r = 0
            
        elif self.current_state == 1: # 1 is s2 in the original game in the paper (figure 3)
            if a1 == a2 and a1 == 0:
                r = 1
                self.current_state = 1
                
            elif a1 == a2 and a1 == 1:
                self.current_state = 0
                r = 0
            
            else: # a1 != a2
                r = 0
                self.current_state = 1
                
        else:
            assert False, "There is an error with the state "
            
            
        next_state =  (int(self.current_state), int(self.current_state))
        info = {}
        
        return next_state,  r, done ,info 
        
    def reset(self):
        self.t = 0
        self.current_state = 1 # state 2
        
        return (int(self.current_state), int(self.current_state))
    
    