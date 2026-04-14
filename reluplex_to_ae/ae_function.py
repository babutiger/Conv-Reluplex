# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p

folder = p.file_base + "/reluplex_to_ae/s5_ae_parameter_temp/"
folder_divided = p.file_base + "/reluplex_to_ae/s5_ae_divided_parameter_temp/"


# for one map version
def divided_layer2_pool_all(data, i):
    lst = []
    for layer_1 in data:
        for layer_2 in layer_1:
            item = layer_2[i]
            lst.append(item)
    result = np.reshape(np.array(lst), (p.layer2_pool_result_size,p.layer2_pool_result_size,1))
    return result


def divided_layer4_pool_all(data, i):
    lst = []
    for layer_1 in data:
        for layer_2 in layer_1:
            item = layer_2[i]
            lst.append(item)
    result = np.reshape(np.array(lst), (p.layer4_pool_result_size,p.layer4_pool_result_size,1)) # 4,4,1
    return result


def find_max(data_0, data_1, data_2, data_3 ):
    max_value = data_0
    if  (data_1 > max_value):
        max_value = data_1
    if  (data_2 > max_value):
        max_value = data_2
    if  (data_3 > max_value):
        max_value = data_3
    return max_value

def reverse_layer2_pool_to_layer1_after_relu(layer2_pool_i, layer1_after_relu_i):

    layer2_pool_i = layer2_pool_i.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size) # 4,4
    layer1_after_relu_i = layer1_after_relu_i.reshape(p.layer1_conv_result_size, p.layer1_conv_result_size) # 8,8

    len_a = len(layer2_pool_i)  # 4
    for i in range(len_a):
        len_b = len(layer2_pool_i[i])  # 8
        for j in range(len_b):
            curr_pool_result = layer2_pool_i[i][j]
            y_start = i * 2
            y_end = i * 2 + 1
            x_start = j * 2
            x_end = j * 2 + 1

            max_value = layer1_after_relu_i[y_start][x_start]
            max_value_y = y_start
            max_value_x = x_start
            if (layer1_after_relu_i[y_start][x_end] > max_value):
                max_value = layer1_after_relu_i[y_start][x_end]
                max_value_y = y_start
                max_value_x = x_end
            if (layer1_after_relu_i[y_end][x_start] > max_value):
                max_value = layer1_after_relu_i[y_end][x_start]
                max_value_y = y_end
                max_value_x = x_start
            if (layer1_after_relu_i[y_end][x_end] > max_value):
                max_value = layer1_after_relu_i[y_end][x_end]
                max_value_y = y_end
                max_value_x = x_end

            if  (curr_pool_result > max_value ):
                layer1_after_relu_i[max_value_y][max_value_x] = curr_pool_result

            if  (curr_pool_result < max_value):
                if (layer1_after_relu_i[y_start][x_start] > curr_pool_result):
                    layer1_after_relu_i[y_start][x_start] = curr_pool_result

                if (layer1_after_relu_i[y_start][x_end] > curr_pool_result):
                    layer1_after_relu_i[y_start][x_end] = curr_pool_result

                if (layer1_after_relu_i[y_end][x_start] > curr_pool_result):
                    layer1_after_relu_i[y_end][x_start] = curr_pool_result

                if (layer1_after_relu_i[y_end][x_end] > curr_pool_result):
                    layer1_after_relu_i[y_end][x_end] = curr_pool_result

    return layer1_after_relu_i.reshape(p.layer1_conv_result_size, p.layer1_conv_result_size, 1)

def reverse_layer4_pool_to_layer3_after_relu(layer4_pool_i, layer3_after_relu_i):

    layer4_pool_i = layer4_pool_i.reshape(p.layer4_pool_result_size, p.layer4_pool_result_size) # 4,4
    layer3_after_relu_i = layer3_after_relu_i.reshape(p.layer3_conv_result_size, p.layer3_conv_result_size) # 8,8

    len_a = len(layer4_pool_i)  # 4
    for i in range(len_a):
        len_b = len(layer4_pool_i[i])  # 8
        for j in range(len_b):
            curr_pool_result = layer4_pool_i[i][j]
            y_start = i * 2
            y_end = i * 2 + 1
            x_start = j * 2
            x_end = j * 2 + 1

            max_value = layer3_after_relu_i[y_start][x_start]
            max_value_y = y_start
            max_value_x = x_start
            if (layer3_after_relu_i[y_start][x_end] > max_value):
                max_value = layer3_after_relu_i[y_start][x_end]
                max_value_y = y_start
                max_value_x = x_end
            if (layer3_after_relu_i[y_end][x_start] > max_value):
                max_value = layer3_after_relu_i[y_end][x_start]
                max_value_y = y_end
                max_value_x = x_start
            if (layer3_after_relu_i[y_end][x_end] > max_value):
                max_value = layer3_after_relu_i[y_end][x_end]
                max_value_y = y_end
                max_value_x = x_end

            if  (curr_pool_result > max_value ):
                layer3_after_relu_i[max_value_y][max_value_x] = curr_pool_result

            if  (curr_pool_result < max_value):
                if (layer3_after_relu_i[y_start][x_start] > curr_pool_result):
                    layer3_after_relu_i[y_start][x_start] = curr_pool_result

                if (layer3_after_relu_i[y_start][x_end] > curr_pool_result):
                    layer3_after_relu_i[y_start][x_end] = curr_pool_result

                if (layer3_after_relu_i[y_end][x_start] > curr_pool_result):
                    layer3_after_relu_i[y_end][x_start] = curr_pool_result

                if (layer3_after_relu_i[y_end][x_end] > curr_pool_result):
                    layer3_after_relu_i[y_end][x_end] = curr_pool_result

    return layer3_after_relu_i.reshape(p.layer3_conv_result_size, p.layer3_conv_result_size, 1) # 8,8,1

# def reverse_layer3_after_relu_to_layer2_pool(layer3_after_relu_i, layer2_pool_i):





def mkdir_if_not_exit():
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder_divided):
        os.mkdir(folder_divided)


def compute_map_num(data):
    for layer_1 in data:
        for layer_2 in layer_1:
            return len(layer_2)

def feature_map_save_divided_i( filename, i, data):
    curr_folder = folder_divided + filename + "/"
    if not os.path.exists(curr_folder):
        os.mkdir(curr_folder)

    name = curr_folder + filename + "_" + str(i)
    file = open(name, 'w+')
    file.write("[\n")
    for layer_1 in data:
        file.write("\t[")
        for item in layer_1:
            s = '{:<15}'.format(str(item[0]))
            s = "[ " + s + "], "
            file.write(s)
        file.write("\t]\n")
    file.write("\n]")
    file.close()
    print("ae.feature_map_save_divided_i - 保存文件成功")

def feature_map_save_divided( filename, data):
    feature_map_num = compute_map_num(data)
    for i in range(feature_map_num):
        curr_folder = folder_divided + filename + "/"
        if not os.path.exists(curr_folder):
            os.mkdir(curr_folder)

        name = curr_folder + filename + "_" + str(i)
        file = open(name, 'w+')
        file.write("[\n")
        for layer_1 in data:
            file.write("\t[")
            for layer_2 in layer_1:
                item = layer_2[i]
                s = '{:<15}'.format(str(item))
                s = "[ " + s + " ], "
                file.write(s)
            file.write("],\n")
        file.write("]\n")
        file.close()
    print("feature_map_save_divided - 保存文件成功")

def feature_map_save( filename, data, divided_flag):
    mkdir_if_not_exit()
    file = open(folder + filename, 'w+')
    file.write("[\n")
    for layer_1 in data:
        file.write("\t[")
        for layer_2 in layer_1:
            s = ",".join('{:<15}'.format(str(i)) for i in layer_2)
            s = "[ " + s + " ],"
            file.write(s)
        file.write("],\n")
    file.write("]\n")

    file.close()
    print("feature_map_save - 保存文件成功")
    if divided_flag:
        feature_map_save_divided( filename, data)
