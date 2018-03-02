# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:55:31 2018

@author: David
"""

import tensorflow as tf
num = tf.Variable(0,name="count")
new_value = tf.add(num,1)
op = tf.assign(num,new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(num))
    for i in range(5):
        sess.run(op)
        print(sess.run(num))