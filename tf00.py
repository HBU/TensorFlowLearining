# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:48:11 2018

@author: David
"""

import tensorflow as tf

v1 = tf.constant([[2,3]])
v2 = tf.constant([[2],[3]])

product = tf.matmul(v1,v2)
print (product)

sess = tf.Session()
result = sess.run(product)
print (result)

sess.close