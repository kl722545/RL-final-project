import random
import SPEC

# description : quest. location. previous action. mislead (not including quest)
# 4 sentences / 20 words

#home_quest = you are hungry --> room = kitchen, action = eat something
#home_quest = you are sleepy --> room = bedroom, action = sleep
#home_quest = you are bored --> room = livingroom, action = watch TV
#home_quest = you are getting fat --> room = garden, action = do exercise
#home_quest = you are dirty --> room = toilet, action = take bath

class HomeWorld():
    def __init__(self):
        self.total_reward = 0
        self.home_rooms = SPEC.home_rooms
        self.home_actions = SPEC.home_actions
        self.home_objects = SPEC.home_objects
        self.home_quests = SPEC.home_quests
        self.T = SPEC.T
        self.t = 0
    def new_game(self,verbose = True):
        self.verbose = verbose
        if self.verbose:
            print ('The new game is now started')
            print ('Welcome to our Home World!!')
            print (SPEC.intro)
        self.t = 0
        self.total_reward = 0
        self.current_room = random.choice(self.home_rooms)
        self.current_home_action = "do"
        self.current_home_object = "nothing"
        self.get_quest()
        self.get_mislead()
    def get_current_state(self):
        return\
        SPEC.home_quest_seq.format(self.current_home_quest)+\
        SPEC.home_quest_location_seq.format(self.current_room)+\
        SPEC.home_quest_pre_action_seq.format(self.current_home_action,self.current_home_object)+\
        SPEC.home_quest_mislead_seq.format(self.current_home_quest_mislead)
    def if_finished(self):
        if self.t == self.T:
            if self.verbose:
                print("overall_reward : {0}".format(self.total_reward))
            return True
        return False
    def do_action(self,action,object=None):
        self.t += 1
        self.current_home_action = action
        self.current_home_object = object
        success = self.location_transition()
        r = self.reward_function()
        if not success and r < 0:
            if self.current_home_action == "go":
                self.current_home_object = "nowhere"
            else:
                self.current_home_action = "do"
                self.current_home_object = "nothing"
        self.total_reward += r
        self.get_mislead()
        return r
    def location_transition(self):
        find_pos = (self.current_room,self.current_home_action,self.current_home_object)
        if find_pos in SPEC.home_map.keys():
            self.current_room = SPEC.home_map[find_pos]
            return True
        return False
    def reward_function(self):
        find_reward = (self.current_home_quest,self.current_room,self.current_home_action,self.current_home_object)
        if find_reward in SPEC.reward_map.keys():
            r = SPEC.reward_map[find_reward]
            self.get_quest()
            return r
        return -0.01
    def get_quest(self):
        self.index, self.current_home_quest  =  random.choice(list(enumerate(self.home_quests)))
    def get_mislead(self):
        self.current_home_quest_mislead = random.choice([q for i,q in enumerate(self.home_quests) if i!=self.index])    