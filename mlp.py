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

cost = [0.0]*2000
acc_train1=[0.0]*20
acc_test1=[0.0]*20
acc_train2=[0.0]*20
acc_test2=[0.0]*20
acc_train3=[0.0]*20
acc_test3=[0.0]*20

k1 = 0.5 ; k2 = 0.05; k3 = 0.005;
#k= 0.5
#e1 = 0.1; e2 = 0.01; e3=0.001;
e=0.01
#t1=1000; t2=5000; t3=10000;
t=2000

# SGD (1)
for i in range(1,t):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	learning_rate = k1 / i
	c_lr = (1-learning_rate)*e+learning_rate
	sess.run([train,loss],feed_dict={x:batch_xs,y:batch_ys,lr:c_lr})
	if i%100 == 0:
		acc_train1[i/100] = sess.run(accuracy,
									feed_dict={x:batch_xs,y:batch_ys})
		acc_test1[i/100] = sess.run(accuracy,
									 feed_dict={x: mnist.test.images, y: mnist.test.labels})

for i in range(1,t):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	learning_rate = k2 / i
	c_lr = (1 - learning_rate) * e + learning_rate
	sess.run([train, loss], feed_dict={x: batch_xs, y: batch_ys, lr: c_lr})
	if i % 100 == 0:
		acc_train2[i / 100] = sess.run(accuracy,
									   feed_dict={x: batch_xs, y: batch_ys})
		acc_test2[i / 100] = sess.run(accuracy,
									  feed_dict={x: mnist.test.images, y: mnist.test.labels})

for i in range(1,t):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	learning_rate = k3 / i
	c_lr = (1 - learning_rate) * e + learning_rate
	#c_lr = min(0.15,0.5*pow(0.999,i))
	sess.run([train, loss], feed_dict={x: batch_xs, y: batch_ys, lr: c_lr})
	if i % 100 == 0:
		acc_train3[i / 100] = sess.run(accuracy,
									   feed_dict={x: batch_xs, y: batch_ys})
		acc_test3[i / 100] = sess.run(accuracy,
									  feed_dict={x: mnist.test.images, y: mnist.test.labels})

# evaluation
# correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

print 'train1: ',acc_train1[19],' test1:',acc_test1[19]
print 'train2: ',acc_train2[19],' test2:',acc_test2[19]
print 'train3: ',acc_train3[19],' test3:',acc_test3[19]

## draw
fig = plt.figure()
plt.xlabel(u"Training set Iterations")
plt.ylabel(u"Accuracy")
plt.plot(xrange(20),acc_train1,'ro-',label='acc_train-t1')
plt.plot(xrange(20),acc_test1,'rs-',label='acc_test-t1')
plt.plot(xrange(20),acc_train2,'bo-',label='acc_train-t2')
plt.plot(xrange(20),acc_test2,'bs-',label='acc_test-t2')
plt.plot(xrange(20),acc_train3,'go-',label='acc_train-t3')
plt.plot(xrange(20),acc_test3,'gs-',label='acc_test-t3')
plt.legend(loc='lower right')
plt.show()

# plt.ylabel(u"MSE")
# plt.xlabel(u"Training set Iterations")
# plt.plot(xrange(2000),cost1,'b')
# plt.legend(loc='lower right')
# plt.show()