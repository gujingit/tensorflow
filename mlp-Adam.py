#coding=utf-8
import tensorflow as tf
import input_data
import math
import matplotlib.pyplot as plt
#参考博客 http://blog.csdn.net/mao_xiao_feng/article/details/53382790 tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)
#http://blog.csdn.net/helei001/article/details/51697658 整体流程
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

#ReLU
l1 = tf.nn.relu(tf.matmul(x,w0)+b0)
l2 = tf.nn.relu(tf.matmul(l1,w1)+b1)
l3 = tf.nn.relu(tf.matmul(l2,w2)+b2)
output = l3

#loss = tf.reduce_mean(tf.pow((output-y),2.0)) #使用欧氏距离,效果很差
#先对最后一层做一个softmax,然后计算交叉熵,取均值作为loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output)) #logits 神经网络最后一层输出 labels实际标签

train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

tmpT = 2000
tmpIndex = tmpT/100
cost = [0.0]*tmpT
acc_train1=[0.0]*tmpIndex
acc_test1=[0.0]*tmpIndex
acc_train2=[0.0]*tmpIndex
acc_test2=[0.0]*tmpIndex
acc_train3=[0.0]*tmpIndex
acc_test3=[0.0]*tmpIndex

iter_time=tmpT

learning_rate1 = 0.001
learning_rate2 = 0.0001
learning_rate3 = 0.00001


for i in xrange(iter_time):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	t,c=sess.run([train,loss],feed_dict={x:batch_xs,y:batch_ys,lr:learning_rate1})
	if i%100 == 0:
		acc_train1[i/100] = sess.run(accuracy,
									feed_dict={x:batch_xs,y:batch_ys})
		acc_test1[i/100] = sess.run(accuracy,
									 feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print "cost1: ",c

for i in xrange(iter_time):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	t,c=sess.run([train, loss], feed_dict={x: batch_xs, y: batch_ys, lr: learning_rate2})
	if i % 100 == 0:
		acc_train2[i / 100] = sess.run(accuracy,
									   feed_dict={x: batch_xs, y: batch_ys})
		acc_test2[i / 100] = sess.run(accuracy,
									  feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print "cost2", i / 10, ": ", c

for i in xrange(iter_time):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	t,c = sess.run([train, loss], feed_dict={x: batch_xs, y: batch_ys, lr: learning_rate3})
	if i % 100 == 0:
		acc_train3[i / 100] = sess.run(accuracy,
									   feed_dict={x: batch_xs, y: batch_ys})
		acc_test3[i / 100] = sess.run(accuracy,
									  feed_dict={x: mnist.test.images, y: mnist.test.labels})
		print "cost3", i / 10, ": ", c

# evaluation
# correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

print 'train1: ',acc_train1[tmpIndex-1],' test1:',acc_test1[tmpIndex-1]
print 'train2: ',acc_train2[tmpIndex-1],' test2:',acc_test2[tmpIndex-1]
print 'train3: ',acc_train3[tmpIndex-1],' test3:',acc_test3[tmpIndex-1]

## draw
fig = plt.figure()
plt.xlabel(u"Training set Iterations")
plt.ylabel(u"Accuracy")
plt.plot(xrange(tmpIndex),acc_train1,'bo-',label='acc_train_Adam1')
plt.plot(xrange(tmpIndex),acc_test1,'bs-',label='acc_test_Adam1')
plt.plot(xrange(tmpIndex),acc_train2,'ro-',label='acc_train_Adam2')
plt.plot(xrange(tmpIndex),acc_test2,'rs-',label='acc_test_Adam2')
plt.plot(xrange(tmpIndex),acc_train3,'go-',label='acc_train_Adam3')
plt.plot(xrange(tmpIndex),acc_test3,'gs-',label='acc_test_Adam3')
plt.legend(loc='lower right')
plt.show()

# plt.ylabel(u"MSE")
# plt.xlabel(u"Training set Iterations")
# plt.plot(xrange(2000),cost1,'b')
# plt.legend(loc='lower right')
# plt.show()