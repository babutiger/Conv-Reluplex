# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p


path = p.file_base + "transform_nnet_parameter/s3_param_file/"


def build_folder():
    if not os.path.exists(path):
        os.mkdir(path)

def build_folder(path_sepical):
    if not os.path.exists(path_sepical):
        os.mkdir(path_sepical)


def transform_input_special_name(weight, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")
    for layer_1 in weight:
        for layer_2 in layer_1:
            for layer_3 in layer_2:
                s = ",".join(str(i) for i in layer_3)
                s = s + " ,"
                file.write(s)
    file.close()

def transform_input(weight, name):
    build_folder()
    file = open(path + name, "w+")
    for layer_1 in weight:
        for layer_2 in layer_1:
            for layer_3 in layer_2:
                s = ",".join(str(i) for i in layer_3)
                s = s + " ,"
                file.write(s)
    file.close()


def transform_weight_special_name_line(weight, original_row, original_col, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")
    result = []
    for i in range(original_col):
        arr_row = []
        for j in range(original_row):
            arr_row.append(weight[j][i])
        s = ",".join(str(i) for i in arr_row)
        s = s + ","
        file.write(s)
    file.close()
    return result

def transform_weight_special_name(weight, original_row, original_col, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")
    result = []
    for i in range(original_col):
        arr_row = []
        for j in range(original_row):
            arr_row.append(weight[j][i])
        s = ",".join(str(i) for i in arr_row)
        s = s + ",\n"
        file.write(s)
    file.close()
    return result
def transform_weight(weight, original_row, original_col, name):
    build_folder()
    file = open(path + name, "w+")
    result = []
    for i in range(original_col):
        arr_row = []
        for j in range(original_row):
            arr_row.append(weight[j][i])
        s = ",".join(str(i) for i in arr_row)
        s = s + ",\n"
        file.write(s)
    file.close()
    return result


def transform_biases_special_name(biases, name, path_sepical):
    build_folder(path_sepical)
    file = open(path_sepical + name, "w+")
    for item in biases:
        s = str(item) + ",\n"
        file.write(s)
    file.close()

def transform_biases(biases, name):
    build_folder()
    file = open(path + name, "w+")
    for item in biases:
        s = str(item) + ",\n"
        file.write(s)
    file.close()
