# -*- coding: utf-8 -*-

from skimage import io,transform
import os
import glob
import numpy as np
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import read_parameter as rd
from mycode.mnist_all_minish_one_map_9_9.conv_network_simulation import simulation_function as mfc
from mycode.mnist_all_minish_one_map_9_9.reluplex_to_ae import ae_function as aefc
from mycode.mnist_all_minish_one_map_9_9 import s0_parameter_all as p



# ae_fc_input = p.ae_fc_input
# ae_layer2_pool_all = ae_fc_input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size,
#                                          p.layer1_conv_amount)  # 4 * 4 * 4
# aefc.feature_map_save("ae_layer2_pool", ae_layer2_pool_all, True)
#
#
# ae_layer2_pool_0 = aefc.divided_layer2_pool_all(ae_layer2_pool_all, 0)
#
# original_layer1_after_relu_0 = rd.read_layer1_after_relu_divided("layer1_after_relu_0")
#
#
# ae_layer1_after_relu_0 = aefc.reverse_layer2_pool_to_layer1_after_relu(ae_layer2_pool_0,
#                                                                        original_layer1_after_relu_0)
#
#

def five_step(ae_fc_input_from_p):

    ae_fc_input = ae_fc_input_from_p
    ae_layer2_pool_all = ae_fc_input.reshape(p.layer2_pool_result_size, p.layer2_pool_result_size,
                                             p.layer1_conv_amount)  # 4 * 4 * 4
    aefc.feature_map_save("ae_layer2_pool", ae_layer2_pool_all, True)


    ae_layer2_pool_0 = aefc.divided_layer2_pool_all(ae_layer2_pool_all, 0)

    original_layer1_after_relu_0 = rd.read_layer1_after_relu_divided("layer1_after_relu_0")


    ae_layer1_after_relu_0 = aefc.reverse_layer2_pool_to_layer1_after_relu(ae_layer2_pool_0,
                                                                           original_layer1_after_relu_0)

    aefc.feature_map_save_divided_i("ae_layer1_after_relu", 0, ae_layer1_after_relu_0)
