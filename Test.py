import tensorflow as tf
import numpy as np

R = [1,2,3,4]
print(R[0:2])
print(R[2:4])

w1 = tf.Variable(tf.random_normal([3, 2],dtype=tf.float32))

#initialize the variable
init_op = tf.global_variables_initializer()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(w1))

