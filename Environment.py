import numpy as np
import pandas as pd
import os
import time
import string
import random
import sys

# description : quest. location. previous action. mislead (not including quest)
# 4 sentences / 20 words

description = {}
agent_location = None
agent_quest = []
agent_action_outcome = []
agent_mislead = []
quest_checklist = []


class HomeWorld():
    def __init__(self, num_rooms=4, seq_length=20, max_step=5):
        self.regions = ["home", "school"]
        self.home_rooms = ["livingroom", "garden", "kitchen", "bedroom", "toilet"]
        self.school_rooms = ["physics_classroom", "math_classroom", "music_classroom", "canteen", "field", "library"]
        self.actions = ["eat", "sleep", "watch", "exercise", "go", "study", "bath", "borrow", "attend"]
        self.objects = ["north", "south", "east", "west", "apple", "book", "alone", "book", "class", "body"]
        self.quests = ["You are hungry", "You are sleepy", "You are bored", "You are getting fat",
                       "You are dirty", "You are going to school", "You are going to home",
                       "You are going to borrow books",
                       "You are going to attend math class", "You are going to attend physics class",
                       "You are going to attend music class"]
        self.quests_mislead = ["You are not hungry", "You are not sleepy", "You are not bored",
                               "You are not getting fat",
                               "You are not dirty", "You are not going to school", "You are not going to home",
                               "You are not going to borrow books",
                               "You are not going to attend music class"]
        self.idx2word = ["not", "but", "now"]
        self.x = 'sdfjlsd'

    def get_random_location(self):
        return random.choice(self.home_rooms)

    def read_file(self):
        fp = open("description.txt")
        lines = fp.readlines()
        for room in self.home_rooms:
            if room == "livingroom":
                temp = lines[1] + lines[2]
            elif room == "garden":
                temp = lines[5] + lines[6]
            elif room == "kitchen":
                temp = lines[9] + lines[10] + lines[11]
            elif room == "bedroom":
                temp = lines[14]
            else:
                temp = lines[17]
            description[room] = temp

    def new_game(self):
        self.read_file()
        # self.test()
        print ('Welcome to our Home World!!')
        self.first_location = self.get_random_location()
        print (description[self.first_location])


def check_location(action_list):
    global agent_location
    room = agent_location
    print ('action_list:', action_list)
    print ('room:', room)
    if room == 'livingroom':
        print ('1')
        if action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'garden'
        elif action_list[0] == 'go' and action_list[1] == 'south':
            agent_location = 'bedroom'
        else:
            agent_location = room
    if room == 'garden':
        print ('2')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'livingroom'
        elif action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'kitchen'
        elif action_list[0] == 'go' and action_list[1] == 'south':
            agent_location = 'toilet'
        else:
            agent_location = room
    if room == 'kitchen':
        print ('3')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'garden'
        else:
            agent_location = room
    if room == 'bedroom':
        print ('4')
        if action_list[0] == 'go' and action_list[1] == 'east':
            agent_location = 'toilet'
        elif action_list[0] == 'go' and action_list[1] == 'north':
            agent_location = 'livingroom'
        else:
            agent_location = room
    if room == 'toilet':
        print ('5')
        if action_list[0] == 'go' and action_list[1] == 'west':
            agent_location = 'bedroom'
        elif action_list[0] == 'go' and action_list[1] == 'north':
            agent_location = 'garden'
        else:
            agent_location = room

    print ('agent location inside check: ',agent_location)


def handle_action(action):
    action_list = action.split()
    if len(action_list) == 2:
        check_location(action_list)
    else:
        print ('try to input two words action command')


time_step = 5
RL_environment = HomeWorld()

print ('The game is now started')
RL_environment.new_game()
for i in range(time_step):
    if i == 0:
        agent_location = RL_environment.first_location
    else:
        pass
    input_command = input('What do you want to do? : ')
    print ('agent_location inside for loop: ', agent_location)
    handle_action(input_command)

    # print input_command
