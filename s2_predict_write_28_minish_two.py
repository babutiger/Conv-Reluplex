# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
import pylab
import tensorflow as tf
from mycode.mnist_all_minish_one_map_9_9 import functions as fs
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p
import matplotlib.image as mpimg



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# test_path = "./mnist_predict_write/"
# test_data,test_label = fs.read_image(test_path)

def second_step_special(img_file, test_label):

    test_label = np.array([test_label])
    original_input_x = mpimg.imread(img_file)
    test_data = original_input_x.reshape(1, 28, 28, 1)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(p.model +'/model.ckpt.meta')
        saver.restore(sess, p.model + '/model.ckpt')

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")

        fs.show_image_label(test_data[0], test_label[0])

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
        print("y:", result[17][0])  # this will repeat if predict more than one , so just get the first result
        print("y_:", result[18][0])

        # change to one map

        fs.feature_map_save("x", np.array([result[16][0]]), False)

        fs.weight_save("layer1_conv_weights", np.array([result[0][0]]), True)
        fs.biases_save("layer1_conv_biases", np.array([result[1][0]]))
        fs.feature_map_save("layer1_conv_result", np.array([result[2][0]]), True)
        fs.feature_map_save("layer1_after_relu", np.array([result[3][0]]), True)

        fs.feature_map_save("layer2_pool", np.array([result[4][0]]), True)

        fs.fc_weight_save("fc1_weights", np.array([result[5][0]]))
        fs.biases_save("fc1_biases", np.array([result[6][0]]))
        fs.fc_after_relu_save("fc1_after_relu", np.array([result[7][0]]))

        fs.fc_weight_save("fc2_weights", np.array([result[8][0]]))
        fs.biases_save("fc2_biases", np.array([result[9][0]]))
        fs.fc_after_relu_save("fc2_after_relu", np.array([result[10][0]]))

        fs.fc_weight_save("fc3_weights", np.array([result[11][0]]))
        fs.biases_save("fc3_biases", np.array([result[12][0]]))
        fs.fc_after_relu_save("fc3_result", np.array([result[13][0]]))

        # change end


def second_step():
    original_file = p.original_file
    test_label = p.test_label

    original_input_x = mpimg.imread(original_file)
    test_data = original_input_x.reshape(1,28,28,1)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(p.model + '/model.ckpt.meta')
        saver.restore(sess, p.model + '/model.ckpt')

        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")

        fs.show_image_label(test_data[0], test_label[0])

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
        print("y:", result[17][0])  # this will repeat if predict more than one , so just get the first result
        print("y_:", result[18][0])

        # change to one map

        fs.feature_map_save("x", result[16], False)

        fs.weight_save("layer1_conv_weights", result[0], True)
        fs.biases_save("layer1_conv_biases", result[1])
        fs.feature_map_save("layer1_conv_result", result[2], True)
        fs.feature_map_save("layer1_after_relu", result[3], True)

        fs.feature_map_save("layer2_pool", result[4], True)

        fs.fc_weight_save("fc1_weights", result[5])
        fs.biases_save("fc1_biases", result[6])
        fs.fc_after_relu_save("fc1_after_relu", result[7])

        fs.fc_weight_save("fc2_weights", result[8])
        fs.biases_save("fc2_biases", result[9])
        fs.fc_after_relu_save("fc2_after_relu", result[10])

        fs.fc_weight_save("fc3_weights", result[11])
        fs.biases_save("fc3_biases", result[12])
        fs.fc_after_relu_save("fc3_result", result[13])


        # change end
if __name__ == "__main__":
    second_step()