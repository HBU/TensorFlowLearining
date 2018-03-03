# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:07:39 2018

@author: David
"""
import tensorflow as tf;    
import numpy as np;    
import matplotlib.pyplot as plt;    
  
v1 = tf.Variable(tf.constant(1, shape=[1]), name='v1')  
v2 = tf.Variable(tf.constant(2, shape=[1]), name='v2')  
  
result = v1 + v2  
  
init = tf.initialize_all_variables()  
  
saver = tf.train.Saver()  
  
with tf.Session() as sess:  
    sess.run(init)  
    saver.save(sess, "./model.ckpt")  // 大坑。。。