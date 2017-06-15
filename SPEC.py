import numpy

T = 5
intro = "The home map:\n\
[LivingRoom]-->[Garden]-->[Kitchen]\n\
    |             |\n\
    |             |\n\
    V             V\n\
[Bedroom]   -->[Toilet]\n\
Living Room\n\
You have entered the living room. You can watch TV here.\n\
This room has two sofas, chairs and a TV.\n\
Garden\n\
You have arrived at the garden. You can do exercise here.\n\
This space has a bike, flowers and trees \n\
Kitchen\n\
You have arrived in the kitchen. You can find somthing to eat here.\n\
This living area has pizza, coke, and ice cream.\n\
This room has a fridge, oven, and a sink.\n\
Bedroom\n\
You have arrived in the bedroom. You can sleep here.\n\
Toilet\n\
You have entered the toilet. You can take bath here.\n\n"

# Dictionay all words 
all_words = ['You','are','not','at','in','the','hungry','sleepy','bored','getting', \
             'fat','dirty','going','to','books','class','home','school','living_room','garden', \
             'kitchen','bedroom','toilet','physics','math','music','canteen','field','library','eat', \
             'sleep','watch','go', 'study','take','borrow','attend','north','south','east', \
             'west','something',"bath","TV","classroom","nowhere","do","nothing","."]

vocabulary = len(all_words)

# Vector data dimension
vec_dim = vocabulary + 1
seq_len = 20
seq_num = 4  

# (1) Quest (2) Location (3) Pre-Action (4) Quest-mislead , Reward
# Ex : 'You are hungry.You are in the kitchen.You eat something.You are hungry.'


# "Life of student" setting
regions = ["home", "school"]

home_rooms = ["living_room", "garden", "kitchen", "bedroom", "toilet"]
school_rooms = ["physics", "math", "music", "canteen", "field", "library"]

home_actions = ['go','eat','sleep','watch','do','take']
school_actions =  ['go', "borrow", "attend","eat"]


home_quests = ["hungry", "sleepy", "bored", "getting fat", "dirty"]


school_quests = ["school","home","eat","borrow books","attend math class","attend physics class","attend music class"]

home_quest_seq = "You are {0}.\n"
home_quest_location_seq = "You are in the {0}.\n"
home_quest_pre_action_seq = "You {0} {1}.\n"
home_quest_mislead_seq = "You are not {0}."

school_quest_seq = "You are going to {0}\n"
school_quest_location_seq = ["You are in the {0}.\n", "You are in the {0} classroom.\n"]
school_quest_pre_action_seq = "You {0} {1}.\n"
school_quest_mislead_seq = "You are not going to {0}."


home_objects = ["north", "south", "east", "west", "something","bath","TV","exercise",None]
school_objects = ["north", "south", "east", "west", "class", "books","something",None]

"""
[LivingRoom]-->[Garden]-->[Kitchen]
    |             |
    |             |
    V             V
[Bedroom]   -->[Toilet]
"""
home_map = {("living_room","go","east") : "garden",\
            ("living_room","go","south") : "bedroom",\
            ("bedroom","go","north") : "living_room",\
            ("bedroom","go","east") : "toilet",\
            ("toilet","go","west") : "bedroom",\
            ("toilet","go","north") : "garden",\
            ("garden","go","south") : "toilet",\
            ("garden","go","east") : "kitchen",\
            ("garden","go","west") : "living_room",\
            ("kitchen","go","west") : "garden"}
"""
hungry --> kitchen, eat something
sleepy --> bedroom, sleep None
bored --> livingroom, watch TV
getting fat --> garden, do exercise
dirty --> toilet, take bath
"""
reward_map = {("hungry","kitchen","eat","something") : 3,\
              ("sleepy","bedroom","sleep",None) : 3,\
              ("bored","living_room","watch","TV") : 3,\
              ("getting fat","garden","do","exercise") : 3,\
			  ("getting fat","garden","exercise",None) : 3,\
              ("dirty","toilet","take","bath") : 3}




