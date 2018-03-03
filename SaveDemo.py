# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 02:19:45 2018

@author: David
"""

import tensorflow as tf  
import numpy as np  
  
# save to file  
W = tf.Variable([[1,2,3],[4,5,6]],dtype = tf.float32,name='weight')  
b = tf.Variable([[1,2,3]],dtype = tf.float32,name='biases')  
  
init = tf.initialize_all_variables()  
saver = tf.train.Saver()  
with tf.Session() as sess:  
        sess.run(init)  
        save_path = saver.save(sess,"my_net/save_net.ckpt")  
        print ("save to path:",save_path)  