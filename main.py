
# coding: utf-8

# In[ ]:


import Environment
import Codec
import Agent
import tensorflow as tf
from collections import deque
import random
import SPEC
import numpy as np
import pickle
import csv

# In[ ]:


codec = Codec.Codec()
RL_environment = Environment.HomeWorld()
agent = Agent.Agent()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:


replay_memory = deque(maxlen = 100000)
for _ in range(3000 // SPEC.T):
    RL_environment.new_game(verbose = False)
    new_start = True
    while True:
        if RL_environment.if_finished():
            replay_memory[-1][5] = True
            #print("overall_reward : {0}".format(RL_environment.total_reward))
            break
        elif new_start:
            state = RL_environment.get_current_state()
            coded_state = codec.encode(state)
        else:
            state = next_state
            coded_state = coded_next_state
        action,object = agent.simulate(sess,[coded_state],epsilon = 1)
        action_str,object_str = codec.decode_action(action,object)
        actions = (action_str[0],object_str[0])
        reward = RL_environment.do_action(*tuple(actions))
        next_state = RL_environment.get_current_state()
        coded_next_state = codec.encode(next_state)
        transection = [coded_state,action[0],object[0],reward,coded_next_state,False]
        replay_memory.append(transection)  


# In[ ]:


saver = tf.train.Saver()
i = 0
epsilon = 1
all_rewards = 0
batch_size = 100
max_iter = 100000
reward_list = [0] * max_iter
for epoch in range(max_iter):
    epsilon *= 0.999995
    RL_environment.new_game(verbose = False)
    new_start = True
    while True:
        i = (i+1) % 4
        if RL_environment.if_finished():
            replay_memory[-1][5] = True
            all_rewards += RL_environment.total_reward
            if epoch % 10 == 0:
                print("epoch:{0:3} overall_reward : {1}".format(epoch,all_rewards / 10))
                all_rewards = 0
            reward_list[epoch] = RL_environment.total_reward
            break
        elif new_start:
            state = RL_environment.get_current_state()
            coded_state = codec.encode(state)
        else:
            state = next_state
            coded_state = coded_next_state
        action,object = agent.simulate(sess,[coded_state],epsilon = epsilon)
        action_str,object_str = codec.decode_action(action,object)
        actions = (action_str[0],object_str[0])
        reward = RL_environment.do_action(*tuple(actions))
        next_state = RL_environment.get_current_state()
        coded_next_state = codec.encode(next_state)
        transections = np.array(random.sample(replay_memory,batch_size))
        minibatch = tuple([np.array(transections[:,i].tolist()) for i in range(6)])
        loss = agent.training(sess,minibatch)
        if i == 0:
            agent.copy_target_Q(sess)
        transection = [coded_state,action[0],object[0],reward,coded_next_state,False]
        replay_memory.append(transection)
    if epoch % 1000 == 9:
        saver.save(sess, "./model.ckpt")
        print("model save at {0}".format("./model.ckpt"))


# In[ ]:


for i in range(10):
    RL_environment.new_game(verbose = False)
    new_start = True
    while True:
        if RL_environment.if_finished():
            replay_memory[-1][5] = True
            print("test {0:2} overall_reward : {1}".format(i,RL_environment.total_reward))
            break
        elif new_start:
            state = RL_environment.get_current_state()
            coded_state = codec.encode(state)
        else:
            state = next_state
            coded_state = coded_next_state
        action,object = agent.simulate(sess,[coded_state],epsilon = 0)
        action_str,object_str = codec.decode_action(action,object)
        actions = (action_str[0],object_str[0])
        reward = RL_environment.do_action(*tuple(actions))
        next_state = RL_environment.get_current_state()
        coded_next_state = codec.encode(next_state)


# In[ ]:

with open("reward.dat","wb") as fp:
    pickle.dump(reward_list,fp,protocol=pickle.HIGHEST_PROTOCOL)
sess.close()

"""
with open("reward.csv", 'w') as fp:
    wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
    wr.writerow(reward_list)
"""