
# coding: utf-8

# In[4]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

import tensorflow as tf


# 参数
learning_rate = 0.01
epochs = 1
batch_size = 512


# 用来验证和计算准确率的样本数
test_valid_size = 256

# Network Parameters
# 神经网络参数
n_classes = 10  
dropout = 0.75  
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}


####################################################################################################################
 
    ###完成函数  返回卷积的结果 ,以及最大池化的函数

####################################################################################################################


def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x,W,strides=[1,strides,strides,1], padding='SAME')
    
    x = tf.nn.bias_add(x,b)
    
    return tf.nn.relu(x)



def maxpool2d(x, k=2):
    
    return tf.nn.max_pool(
    x,
    ksize=[1,k,k,1],
    strides=[1,k,k,1],
    padding='SAME' )



######################################################################################################################


def conv_net(x, weights, biases, dropout):
    # 第一层卷积 1 - 28*28*1 to 14*14*16
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    # 卷积层2 - 14*14*16 to 14*14*32
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=1)
    
    # 卷积层3 - 14*14*32 to 7*7*64
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool2d(conv3, k=2)

    # 全连接层 - 7*7*64 to 1024
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # 输出分类 - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out




######################################################################################################################
#######################完成 特征x ，y ，keep_prob的占位符定义
# tf Graph 输入
x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)




###########################################################################################################################

# 模型logits
logits = conv_net(x, weights, biases, keep_prob)

# 损失和优化
cost = tf.reduce_mean(    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    .minimize(cost)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化变量
init = tf. global_variables_initializer()

# 启动会话
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            
            
            
            
            
            ####################################################
            ################完成feed_dict##############占位符定义
            sess.run(optimizer, feed_dict={
                    x:batch_x,
                    y:batch_y,
                    keep_prob:dropout})
            
            ######################################################
            
            

            # 计算batch loss 和准确度 accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # 测试准确度
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
 

 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



