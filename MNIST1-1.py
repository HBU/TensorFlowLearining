# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:31:14 2018

@author: David
"""

import numpy as np
import pandas as pd
import tensorflow as tf

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    saver.restore(sess,"model-1.ckpt-19")
    
    test_x = np.array(test,dtype = np.float32)
    
    conv_y_preditct = y_conv.eval(feed_dict={x:test_x[1:100,:],keep_prob:1.0})
    
    conv_y_preditct_all = list()
    for i in np.arange(100,28001,100):
         conv_y_preditct = y_conv.eval(feed_dict={x:test_x[1:100,:],keep_prob:1.0})
         test_pred= np.argmax(conv_y_preditct,axis = 1)
         conv_y_preditct_all= np.append(conv_y_preditct_all,test_pred)
         
    submission = pd.DataFrame({"ImageId":range(1,28001),"Label":np.int32(conv_y_preditct_all)})
    submission.to_csv("submission.csv",index = False)
    