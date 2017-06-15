import tensorflow as tf
import numpy as np
import One_hot_encoding
import Environment
class Agent:
    def __init__ (self,world):
        self.vec_dim = One_hot_encoding.vec_dim
        self.max_length = One_hot_encoding.max_length
        self.hidden_node1 = 15
        self.hidden_node2 = 15
        self.world = world
        self.action_num = len(self.world.actions) 
        self.object_num = len(self.world.objects) 
        initializer = tf.contrib.layers.xavier_initializer()
        #input
        self.w = tf.placeholder(tf.float32, [None, self.max_length , self.vec_dim])
        self.w_next = tf.placeholder(tf.float32, [None, self.max_length , self.vec_dim])
        self.a = tf.placeholder(tf.int32, [None, self.max_length , self.vec_dim])
        self.o = tf.placeholder(tf.int32, [None, self.max_length , self.vec_dim])
        self.r = tf.placeholder(tf.float32, [None])
        self.epsilon = tf.placeholder(tf.float32, [None])
        self.if_ternimal = tf.placeholder(tf.float32, [None])
        self.gamma = 0.99
        with tf.variable_scope("Q_network"):
            self.Qsa, self.Qso = self.Q_network(self.w)
        with tf.variable_scope("target_Q_network"):
            self.target_Qsa_next, self.target_Qso_next = self.Q_network(self.w_next)
        self.define_training_param()
        
    def Q_network(self):
        with tf.variable_scope("Representation_Generator"):
            x_s, _ = tf.contrib.rnn.dynamic_rnn(tf.contrib.rnn.BasicLSTMCell(self.vec_dim), tf.unstack(self.w, self.max_length, 1), dtype=tf.float32)
            v_s = tf.divide(tf.add_n(x_s),self.max_length)
        with tf.variable_scope("action_scorer"):
            with tf.variable_scope("Linear1"):
                W1 = tf.get_variable("Weight",shape = [self.vec_dim, self.hidden_node1],initializer = initializer)
                b1 = tf.get_variable("Bias",shape = [self.hidden_node1],initializer = initializer)
                h1 = tf.matmul(v_s,W1) + b1
            with tf.variable_scope("Relu2"):
                W2 = tf.get_variable("Weight",shape = [self.hidden_node1, self.hidden_node2],initializer = initializer)
                b2 = tf.get_variable("Bias",shape = [self.hidden_node2],initializer = initializer)
                h2 = tf.nn.relu(tf.matmul(h1,W2) + b2)
            with tf.variable_scope("Linear3a"):
                W3a = tf.get_variable("Weight",shape = [self.hidden_node2, self.action_num],initializer = initializer)
                b3a = tf.get_variable("Bias",shape = [self.action_num],initializer = initializer)
                Qsa = tf.matmul(h2,W3a) + b3a
            with tf.variable_scope("Linear3o"):
                W3o = tf.get_variable("Weight",shape = [self.hidden_node2, self.object_num],initializer = initializer)
                b3o = tf.get_variable("Bias",shape = [self.object_num],initializer = initializer)
                Qso = tf.matmul(h2,W3o) + b3o
        return Qsa,Qso
    def define_training_param(self):
        random_actions = tf.cast(tf.random_uniform([None], maxval = 1) * self.action_num, dtype = tf.int32)
        random_objects = tf.cast(tf.random_uniform([None], maxval = 1) * self.action_num, dtype = tf.int32)
        greedy_actions = tf.argmax(self.Qsa,axis = 1)
        greedy_objects = tf.argmax(self.Qso,axis = 1)
        
        self.actions = tf.case([tf.less(tf.random_uniform([None], maxval = 1),self.epsilon), random_actions], default = greedy_actions)
        self.objects = tf.case([tf.less(tf.random_uniform([None], maxval = 1),self.epsilon), random_objects], default = greedy_objects)
        with tf.variable_scope("loss"):
            ya = self.r + (1 - self.if_ternimal) * self.gamma * tf.reduce_max(self.traget_Qsa_next,axis = 1)
            yo = self.r + (1 - self.if_ternimal) * self.gamma * tf.reduce_max(self.traget_Qso_next,axis = 1)
            self.loss = tf.cast(tf.square(ya - self.Qsa[:,self.a]) + tf.square(yo - self.Qso[:,self.o]), dtype = tf.float32)
        Q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Q_network")
        target_Q_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"target_Q_network")
        with tf.variable_op_scope("copy_network"):
            self.copy_ops = [tf.assign(t,q.value()) for (q, t) in (Q_network_vars, target_Q_network_vars)]
        with tf.variable_op_scope("train_network"):
            self.train = tf.train.AdamOptimizer().minimize(self.loss, var_list = Q_network_vars)
    def simulate(self,sess,state):
        return sess.run([self.actions, self.objects], {self.w : [state]})
    def train(self,sess,minibatch):
        feed_dict = dict(zip([self.w, self.a, self.o, self.r, self.w_next, self.if_ternimal], minibatch))
        _, loss = sess.run([self.train, self.loss], feed_dict = feed_dict)
        return loss
    def copy_target_Q(self,sess):
        sess.run(self.copy_ops)
        
        
#Agent(Environment.HomeWorld())
