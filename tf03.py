# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:11:59 2018

@author: David
"""

import tensorflow as tf
input1= tf.placeholder(tf.float32)
input2= tf.placeholder(tf.float32)

new_value = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(new_value,feed_dict={input1:23.0,input2:11.0}))