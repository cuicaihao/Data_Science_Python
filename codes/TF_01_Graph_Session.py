#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learning TensorFlow :

Understanding the Graph and Session in TF

Created on Wed May  2 12:20:29 2018

@author: caihaocui
"""

#%% load the package and basic variables
import tensorflow as tf

x = tf.Variable(3, name = 'x')
y = tf.Variable(3, name = 'y')
f = x*x*y + y + 2

#%% not a good way
# 1. initilization 
# 2. run session
# 3. close the session
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result) # 32
sess.close()

#%% good 
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
print(result)

#%% better
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
print(result)

#%% 




