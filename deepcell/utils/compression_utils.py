# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for compressing convolutional neural networks. Adapted from:
https://jacobgil.github.io/deeplearning/tensor-decompositions-deep-learning
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from tensorflow.python.keras.models import Model

from deepcell.layers.compression import TuckerConv, SVDTensorProd


def compress_model(input_model):
    """Go through layer by layer and compress the convolutional
    and tensor product layers. Batch normalization layers are
    unaffected.
    """

    # Start a new model
    new_model_inputs = []
    new_model_outputs = []
    tensor_dict = {}

    model_output_names = [out.name for out in list(input_model.output)]

    for i, layer in enumerate(input_model.layers):
        # Check the input/outputs for each layer
        input_names = [inp.name for inp in list(layer.input)]
        output_names = [out.name for out in list(layer.output)]

        # Setup model inputs
        if 'input' in layer.name:
            for input_tensor in list(layer.output):
                new_model_inputs.append(input_tensor)
                tensor_dict[input_tensor.name] = input_tensor
            continue

        # Setup layer inputs
        layer_inputs = [tensor_dict[name] for name in input_names]
        if len(layer_inputs) == 1:
            inpt = layer_inputs[-1]
        else:
            inpt = layer_inputs

        # Determine if the layer is a convolutional
        # or tensor product layer
        if 'conv2d' in layer.name:
            layer_type = 'conv2d'
        elif 'conv3d' in layer.name:
            layer_type = 'conv3d'
        elif 'tensorprod' in layer.name:
            layer_type = 'tensorprod'
        else:
            layer_type = 'other'

        # Compress the layer using either Tucker
        # decomposition or SVD
        if layer_type == 'conv2d':
            x = TuckerConv(
                layer.filters,
                layer.kernel_size,
                input_weights=layer.get_weights(),
                dilation_rate=layer.dilation_rate,
                padding=layer.padding,
                data_format=layer.data_format,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint)(inpt)

        if layer_type == 'conv3d':
            x = TuckerConv(
                layer.filters,
                layer.kernel_size,
                input_weights=layer.get_weights(),
                dilation_rate=layer.dilation_rate,
                padding=layer.padding,
                data_format=layer.data_format,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint)(inpt)

        if layer_type == 'tensorprod':
            x = SVDTensorProd(
                layer.input_dim,
                layer.output_dim,
                tf.rank(inpt),
                input_weights=layer.get_weights(),
                estimate_rank=True,
                data_format=layer.data_format,
                activation=layer.activation,
                use_bias=layer.use_bias,
                kernel_initializer=layer.kernel_initializer,
                bias_initializer=layer.bias_initializer,
                kernel_regularizer=layer.kernel_regularizer,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=layer.activity_regularizer,
                kernel_constraint=layer.kernel_constraint,
                bias_constraint=layer.bias_constraint)(inpt)

        if layer_type == 'other':
            x = layer(inpt)

        # Add the outputs to the tensor dictionary
        for name, output_tensor in zip(output_names, list(x)):
            # Check if this tensor is a model output
            if name in model_output_names:
                new_model_outputs.append(output_tensor)
            tensor_dict[name] = output_tensor

    # Return compressed model
    return Model(new_model_inputs, new_model_outputs)
