#coding=utf-8
import tensorflow as tf
import input_data
import math
import matplotlib.pyplot as plt

#loadDataSet
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Initialize
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
lr = tf.Variable(dtype=tf.float32,initial_value=0.0)
w0 = tf.Variable(tf.truncated_normal([784,300],stddev=1.0/math.sqrt(float(784))))
b0 = tf.Variable(tf.zeros([300]))
w1 = tf.Variable(tf.truncated_normal([300,100],stddev=1.0/math.sqrt(float(300))))
b1 = tf.Variable(tf.zeros([100]))
w2 = tf.Variable(tf.truncated_normal([100,10],stddev=1.0/math.sqrt(float(100))))
b2 = tf.Variable(tf.zeros([10]))

#Normalization
#x = tf.add((x-tf.reduce_min(x))/(tf.reduce_max(x)-tf.reduce_min(x)),0.00001)
#with tf.Session() as sess:
#	print(sess.run(x[1,:],feed_dict={x:mnist.test.images}))

#=====train=====
#Sigmoid
# l1 = tf.nn.sigmoid(tf.matmul(x,w0)+b0)
# l2 = tf.nn.sigmoid(tf.matmul(l1,w1)+b1)
# l3 = tf.nn.sigmoid(tf.matmul(l2,w2)+b2)

#ReLU
l1 = tf.nn.relu(tf.matmul(x,w0)+b0)
l2 = tf.nn.relu(tf.matmul(l1,w1)+b1)
l3 = tf.nn.relu(tf.matmul(l2,w2)+b2)
output = l3

loss = tf.reduce_mean(tf.pow((output-y),2.0))
#loss = -tf.reduce_sum(y*tf.log(output))
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

cost = [0.0]*5000
acc_train=[0.0]*50
acc_test=[0.0]*50

k= 0.05
e=0.01
t=5000
alpha = [1,0.1,0.01,0.001,0.0001]
index = 0
beforeCost = 99999999999
# SGD (1)
for i in range(0,t):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	t,c=sess.run([train,loss],feed_dict={x:batch_xs,y:batch_ys,lr:alpha[index]})
	#print 'index',alpha[index]
	if beforeCost <= c:
		index += 1
		index = min(index,4)
	beforeCost = c

	if i%100 == 0:
		acc_train[i/100] = sess.run(accuracy,
									feed_dict={x:batch_xs,y:batch_ys})
		acc_test[i/100] = sess.run(accuracy,
									 feed_dict={x: mnist.test.images, y: mnist.test.labels})


# evaluation
# correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

print 'train1: ',acc_train[49],' test1:',acc_test[49]


## draw
fig = plt.figure()
plt.xlabel(u"Training set Iterations")
plt.ylabel(u"Accuracy")
plt.plot(xrange(50),acc_train,'bo-',label='acc_train')
plt.plot(xrange(50),acc_test,'bs-',label='acc_test')
plt.legend(loc='lower right')
plt.show()

# plt.ylabel(u"MSE")
# plt.xlabel(u"Training set Iterations")
# plt.plot(xrange(2000),cost1,'b')
# plt.legend(loc='lower right')
# plt.show()