# -*- coding: utf-8 -*-

from skimage import io,transform
import tensorboard
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
import matplotlib.pyplot as plt
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_path = p.mnist_train_path
test_path = p.mnist_test_path

train_data,train_label = fs.read_image(train_path)
test_data,test_label = fs.read_image(test_path)

train_image_num = len(train_data)
train_image_index = np.arange(train_image_num)
np.random.shuffle(train_image_index)
train_data = train_data[train_image_index]
train_label = train_label[train_image_index]

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

x = tf.placeholder(tf.float32, [None, fs.w, fs.h, fs.c], name='x')
y_ = tf.placeholder(tf.int32 ,[None], name='y_')

def inference(input_tensor,train,regularizer):

    tf.set_random_seed(10)

    with tf.variable_scope('layer1-conv1'):
        # 5，5，1，6
        conv1_weights = tf.get_variable('weight', [p.layer1_conv_size, p.layer1_conv_size, fs.c, p.layer1_conv_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[p.layer1_conv_amount],initializer=tf.constant_initializer(0.0))  # 6
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        tf.summary.histogram("layer1_conv_weights", conv1_weights)
        tf.summary.histogram("layer1_conv_biases", conv1_biases)
        tf.add_to_collection('layer1_conv_weights', conv1_weights)
        tf.add_to_collection('layer1_conv_biases', conv1_biases)
        tf.add_to_collection('layer1_conv_result', conv1)
        tf.add_to_collection('layer1_after_relu', relu1)

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

        tf.summary.histogram("layer2_pool", pool1)

        tf.add_to_collection('layer2_pool', pool1)

    # change to one map

    # with tf.variable_scope('layer3-conv2'):
    #     # 5,5,6,4
    #     conv2_weights = tf.get_variable('weight',[p.layer3_conv_size,p.layer3_conv_size,p.layer1_conv_amount,p.layer3_conv_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
    #     conv2_biases = tf.get_variable('bias',[p.layer3_conv_amount],initializer=tf.constant_initializer(0.0))  # 4
    #     conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='VALID')
    #     relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    #     tf.add_to_collection('layer3_conv_weights', conv2_weights)
    #     tf.add_to_collection('layer3_conv_biases', conv2_biases)
    #     tf.add_to_collection('layer3_conv_result', conv2)
    #     tf.add_to_collection('layer3_after_relu', relu2)
    #
    # with tf.variable_scope('layer4-pool2'):
    #     pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    #     tf.add_to_collection('layer4_pool', pool2)

    # pool_shape = pool2.get_shape().as_list()
    # nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    # reshaped = tf.reshape(pool2,[-1,nodes])

    pool_shape = pool1.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(pool1,[-1,nodes])

    # change end

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,p.fc1_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[p.fc1_amount],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)

        tf.summary.histogram("fc1_weights", fc1_weights)
        tf.summary.histogram("fc1_biases", fc1_biases)
        tf.summary.histogram("fc1_after_relu", fc1)

        tf.add_to_collection('fc1_weights', fc1_weights)
        tf.add_to_collection('fc1_biases', fc1_biases)
        tf.add_to_collection('fc1_after_relu', fc1)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight',[p.fc1_amount,p.fc2_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias',[p.fc2_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

        tf.summary.histogram("fc2_weights", fc2_weights)
        tf.summary.histogram("fc2_biases", fc2_biases)
        tf.summary.histogram("fc2_after_relu", fc2)

        tf.add_to_collection('fc2_weights', fc2_weights)
        tf.add_to_collection('fc2_biases', fc2_biases)
        tf.add_to_collection('fc2_after_relu', fc2)


    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.get_variable('weight',[p.fc2_amount,p.fc3_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc3_weights))
        fc3_biases = tf.get_variable('bias',[p.fc3_amount],initializer=tf.truncated_normal_initializer(stddev=0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

        tf.summary.histogram("fc3_weights", fc3_weights)
        tf.summary.histogram("fc3_biases", fc3_biases)
        tf.summary.histogram("fc3_result", logit)

        tf.add_to_collection('fc3_weights', fc3_weights)
        tf.add_to_collection('fc3_biases', fc3_biases)
        tf.add_to_collection('fc3_result', logit)

    return logit

regularizer = tf.contrib.layers.l2_regularizer(0.001)

y = inference(x, False, regularizer)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_, name="cross_entropy")
cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.int32), y_, name='correct_prediction')
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name='accuracy')

tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)

tf.add_to_collection('x', x)
tf.add_to_collection('y', y)
tf.add_to_collection('y_', y_)
tf.add_to_collection('loss', loss)
tf.add_to_collection('train_op', train_op)

Loss_list_train = []
Accuracy_list_train = []

Loss_list_test = []
Accuracy_list_test = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    p.ensure_dir(p.show_number_model_temp_summary_dir)
    writer_train = tf.summary.FileWriter(
        os.path.join(p.show_number_model_temp_summary_dir, "train"), sess.graph
    )
    writer_test = tf.summary.FileWriter(
        os.path.join(p.show_number_model_temp_summary_dir, "test"), sess.graph
    )


    train_num = 10
    batch_size = 64

    count_train = 0
    count_test = 0

    for i in range(train_num):

        train_loss,train_acc,batch_num,merged_summary = 0, 0, 0, 0
        for train_data_batch,train_label_batch in fs.get_batch(train_data,train_label,batch_size):
            _,err,acc,merged_summary = sess.run([train_op,loss,accuracy,merged],feed_dict={x:train_data_batch,y_:train_label_batch})
            train_loss+=err;train_acc+=acc;batch_num+=1
            count_train += 1


        print("in turn : ", i)
        print("train loss:",train_loss/batch_num)
        print("train acc:",train_acc/batch_num)

        Loss_list_train.append(train_loss/batch_num)
        Accuracy_list_train.append(train_acc/batch_num)

        writer_train.add_summary(merged_summary, i)


        test_loss,test_acc,batch_num,merged_test = 0, 0, 0, 0
        for test_data_batch,test_label_batch in fs.get_batch(test_data,test_label,batch_size):
            err,acc,merged_test = sess.run([loss,accuracy,merged],feed_dict={x:test_data_batch,y_:test_label_batch})
            test_loss+=err;test_acc+=acc;batch_num+=1
            count_test += 1


        print("in turn : ", i)
        print("test loss:",test_loss/batch_num)
        print("test acc:",test_acc/batch_num)

        Loss_list_test.append(test_loss/batch_num)
        Accuracy_list_test.append(test_acc/batch_num)

        writer_test.add_summary(merged_test, i)

    folder = p.show_number_model_temp_model_dir
    p.ensure_dir(folder)

    saver.save(sess, os.path.join(folder, "model.ckpt"))


writer_train.close()
writer_test.close()

# draw


fig,ax = plt.subplots()
ax.set_xlim([0,11])

x1 = range(1, 11)
x2 = range(1, 11)
x3 = range(1, 11)
x4 = range(1, 11)

y1 = Accuracy_list_train
y2 = Loss_list_train
y3 = Accuracy_list_test
y4 = Loss_list_test

p.ensure_dir(p.show_number_model_graph_dir)

plt.plot(x1, y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Train accuracy')
plt.savefig(os.path.join(p.show_number_model_graph_dir, "Train-accuracy.jpg"))
plt.show()

plt.plot(x2, y2, 'g*-')
plt.title('Train loss vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Train loss')
plt.savefig(os.path.join(p.show_number_model_graph_dir, "Train-loss.jpg"))
plt.show()

plt.plot(x3, y3, 'o-', color='orange')
plt.title('Test accuracy vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Test accuracy')
plt.savefig(os.path.join(p.show_number_model_graph_dir, "Test-accuracy.jpg"))
plt.show()

plt.plot(x4, y4, 'g*-', color='orange')
plt.title('Test loss vs. epoches')
plt.xlabel('Epoches')
plt.ylabel('Test loss')
plt.savefig(os.path.join(p.show_number_model_graph_dir, "Test-loss.jpg"))
plt.show()

plt.plot(x1, y1, 'o-', label="Train accuracy")
plt.plot(x1, y3, 'o-', label="Test accuracy", color='orange')
plt.title('Train accuracy vs. Test accuracy')
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(
    os.path.join(p.show_number_model_graph_dir, "Train-accuracy-vs-Test-accuracy.jpg")
)
plt.show()

plt.plot(x2, y2, 'g*-', label="Train loss")
plt.plot(x2, y4, 'g*-', label="Test loss", color='orange')
plt.title('Train loss vs. Test loss')
plt.xlabel('Epoches')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(p.show_number_model_graph_dir, "Train-loss-vs-Test-loss.jpg"))
plt.show()




# plt.plot(x1, y1, 'o-')
# plt.title('Train accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train accuracy')
#
# plt.subplot(2, 3, 2)
# plt.plot(x2, y2, 'g*-')
# plt.title('Train loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Train loss')
#
# plt.subplot(2, 3, 3)
# plt.plot(x3, y3, 'o-', color='orange')
# plt.title('Test accuracy vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test accuracy')
#
# plt.subplot(2, 3, 4)
# plt.plot(x4, y4, 'g*-', color='orange')
# plt.title('Test loss vs. epoches')
# plt.xlabel('Epoches')
# plt.ylabel('Test loss')
#
# plt.subplot(2, 3, 5)
# plt.plot(x1, y1, 'o-')
# plt.plot(x1, y3, 'o-', color='orange')
# plt.title('Train accuracy vs. Test accuracy')
# plt.xlabel('Epoches')
# plt.ylabel('Accuracy')
#
# plt.subplot(2, 3, 6)
# plt.plot(x2, y2, 'g*-')
# plt.plot(x2, y4, 'g*-', color='orange')
# plt.title('Train loss vs. Test loss')
# plt.xlabel('Epoches')
# plt.ylabel('Loss')
#
#
# plt.savefig("./graph/accuracy_loss.jpg")
# plt.show()
