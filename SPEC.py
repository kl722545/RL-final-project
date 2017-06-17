import numpy

T = 50
intro = "The home map:\n\
[LivingRoom]----[Garden]----[Kitchen]\n\
    |             |\n\
    |             |\n\
    |             |\n\
[Bedroom]-------[Toilet]\n\
Living Room\n\
You can watch TV here.\n\
Garden\n\
You can do exercise here.\n\
Kitchen\n\
You can find somthing to eat here.\n\
Bedroom\n\
You can sleep here.\n\
Toilet\n\
You can take bath here.\n\n"

# Dictionay all words 
all_words = [" ",(0,),'You','are','not','at','in','the','hungry','sleepy','bored','getting', \
             'fat','dirty','going','to','books','class','home','school','living_room','garden', \
             'kitchen','bedroom','toilet','physics','math','music','canteen','field','library','eat', \
             'sleep','watch','go', 'study','take','borrow','attend','north','south','east', \
             'west','something',"bath","TV","classroom","exercise","here","nowhere","do","nothing","."]

vocabulary = len(all_words)

# Vector data dimension
vec_dim = vocabulary
seq_len = 9
seq_num = 4 




# "Life of student" setting
regions = ["home", "school"]

home_rooms = ["living_room", "garden", "kitchen", "bedroom", "toilet"]
school_rooms = ["physics", "math", "music", "canteen", "field", "library"]

home_actions = ['go','eat','sleep','watch','do','take']
school_actions =  ['go', "borrow", "attend","eat"]


home_quests = ["hungry", "sleepy", "bored", "getting fat", "dirty"]


school_quests = ["school","home","eat","borrow books","attend math class","attend physics class","attend music class"]


# (1) Quest (2) Location (3) Pre-Action (4) Quest-mislead , Reward
# Ex : 'You are hungry.You are in the kitchen.You eat something.You are hungry.'
home_quest_seq = "You are {0}.\n"
home_quest_location_seq = "You are in the {0}.\n"
home_quest_pre_action_seq = "You {0} {1}.\n"
home_quest_mislead_seq = "You are not {0}.\n"

school_quest_seq = "You are going to {0}\n"
school_quest_location_seq = ["You are in the {0}.\n", "You are in the {0} classroom.\n"]
school_quest_pre_action_seq = "You {0} {1}.\n"
school_quest_mislead_seq = "You are not going to {0}.\n"


home_objects = ["north", "south", "east", "west", "something","bath","TV","exercise","here"]
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
sleepy --> bedroom, sleep here
bored --> livingroom, watch TV
getting fat --> garden, do exercise
dirty --> toilet, take bath
"""
reward_map = {("hungry","kitchen","eat","something") : 3,\
              ("sleepy","bedroom","sleep",None) : 3,\
              ("sleepy","bedroom","sleep","here") : 3,\
              ("bored","living_room","watch","TV") : 3,\
              ("getting fat","garden","do","exercise") : 3,\
              ("getting fat","garden","exercise",None) : 3,\
              ("getting fat","garden","exercise","here") : 3,\
              ("dirty","toilet","take","bath") : 3}




