# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
import pylab
import time
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import matplotlib.image as mpimg



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_path = p.mnist_test_path

test_data,test_label = fs.read_image(test_path)

test_image_num = len(test_data)
test_image_index = np.arange(test_image_num)
np.random.shuffle(test_image_index)
test_data = test_data[test_image_index]
test_label = test_label[test_image_index]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(p.file_base + 'model_9_9/model.ckpt.meta')
    saver.restore(sess, p.file_base + 'model_9_9/model.ckpt')

    # saver = tf.train.import_meta_graph(p.file_base + 'show_number_model/temp-model/model.ckpt.meta')
    # saver.restore(sess, p.file_base + 'show_number_model/temp-model/model.ckpt')

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")

    # fs.show_image_label(test_data[0], test_label[0])

    feed_dict = {x: test_data, y_: test_label}

    layer1_conv_weights = tf.get_collection('layer1_conv_weights')
    layer1_conv_biases = tf.get_collection('layer1_conv_biases')
    layer1_conv_result = tf.get_collection('layer1_conv_result')
    layer1_after_relu = tf.get_collection('layer1_after_relu')

    layer2_pool = tf.get_collection('layer2_pool')

    # layer3_conv_weights = tf.get_collection('layer3_conv_weights')
    # layer3_conv_biases = tf.get_collection('layer3_conv_biases')
    # layer3_conv_result = tf.get_collection('layer3_conv_result')
    # layer3_after_relu = tf.get_collection('layer3_after_relu')
    #
    # layer4_pool = tf.get_collection('layer4_pool')

    fc1_weights = tf.get_collection('fc1_weights')
    fc1_biases = tf.get_collection('fc1_biases')
    fc1_after_relu = tf.get_collection('fc1_after_relu')

    fc2_weights = tf.get_collection('fc2_weights')
    fc2_biases = tf.get_collection('fc2_biases')
    fc2_after_relu = tf.get_collection('fc2_after_relu')

    fc3_weights = tf.get_collection('fc3_weights')
    fc3_biases = tf.get_collection('fc3_biases')
    fc3_result = tf.get_collection('fc3_result')

    x = tf.get_collection('x')
    y = tf.get_collection('y')
    y_ = tf.get_collection('y_')
    accuracy = graph.get_tensor_by_name("accuracy:0")
    correct_prediction = graph.get_tensor_by_name("correct_prediction:0")

    # Add more to the current graph
    # add_on_op = tf.multiply(op_to_restore, 2)

    result = sess.run([layer1_conv_weights, layer1_conv_biases, layer1_conv_result, layer1_after_relu,
                       layer2_pool,
                       # layer3_conv_weights, layer3_conv_biases, layer3_conv_result, layer3_after_relu,
                       # layer4_pool,
                       fc1_weights, fc1_biases, fc1_after_relu,
                       fc2_weights, fc2_biases, fc2_after_relu,
                       fc3_weights, fc3_biases, fc3_result,
                       correct_prediction, accuracy, x, y, y_],
                      feed_dict)
    # print(result)
    print("\n")
    print("correct_prediction:", result[14])
    print("accuracy:", result[15])

    p.ensure_dir(p.adversarial_train_logs_dir)
    log_file = open(
        os.path.join(
            p.adversarial_train_logs_dir,
            "ss1_test_original_use_OR-net" + str(int(round(time.time() * 1000))) + ".txt",
        ),
        "w",
    )
    log_file.write("\ncorrect_prediction: " + str(result[14]))
    log_file.write("\naccuracy: " + str(result[15]))
    log_file.close()
