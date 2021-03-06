#coding=utf-8
import input_data
import tensorflow as tf
import math

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#28*28=784 
#None:no limites in the number of examples
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#=====train=====
#tf.matmul means X*W
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder("float",[None,10])

# cost function
# reduce_sum(a) : sum of tensor a
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs,batch_ys = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#evaluation
# argmax(value,index):return the highest entry of the given index
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

