"""functions used to construct different architectures
"""

import tensorflow as tf
import numpy as np

weight_decay = 0.0005
def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd,initializer=tf.contrib.layers.xavier_initializer()):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    #var = _variable_on_cpu(name, shape,tf.truncated_normal_initializer(stddev = stddev))
    var = _variable_on_cpu(name, shape, initializer)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        weight_decay.set_shape([])
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_layer(inputs, kernel_size, stride, num_features, idx, initializer=tf.contrib.layers.xavier_initializer() , activate="relu"):
    print("conv_layer activation function",activate)
    
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
 
        input_channels = inputs.get_shape()[-1]
        weights = _variable_with_weight_decay('weights',shape = [kernel_size, kernel_size, 
                                                                 input_channels, num_features],
                                              stddev = 0.01, wd = weight_decay)
        biases = _variable_on_cpu('biases', [num_features], initializer)
        conv = tf.nn.conv2d(inputs, weights, strides = [1, stride, stride, 1], padding = 'SAME')
        conv_biased = tf.nn.bias_add(conv, biases)
        if activate == "linear":
            return conv_biased
        elif activate == "relu":
            conv_rect = tf.nn.relu(conv_biased, name = '{0}_conv'.format(idx))  
        elif activate == "elu":
            conv_rect = tf.nn.elu(conv_biased, name = '{0}_conv'.format(idx))   
        elif activate == "leaky_relu":
            conv_rect = tf.nn.leaky_relu(conv_biased, name = '{0}_conv'.format(idx))
        else:
            raise ("activation function is not correct")
        return conv_rect


def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, initializer=tf.contrib.layers.xavier_initializer(),activate="relu"):
    with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
        input_channels = inputs.get_shape()[3]
        input_shape = inputs.get_shape().as_list()


        weights = _variable_with_weight_decay('weights',
                                              shape = [kernel_size, kernel_size, num_features, input_channels],
                                              stddev = 0.1, wd = weight_decay)
        biases = _variable_on_cpu('biases', [num_features],initializer)
        batch_size = tf.shape(inputs)[0]

        output_shape = tf.stack(
            [tf.shape(inputs)[0], input_shape[1] * stride, input_shape[2] * stride, num_features])
        print ("output_shape",output_shape)
        conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides = [1, stride, stride, 1], padding = 'SAME')
        conv_biased = tf.nn.bias_add(conv, biases)
        if activate == "linear":
            return conv_biased
        elif activate == "elu":
            return tf.nn.elu(conv_biased, name = '{0}_transpose_conv'.format(idx))       
        elif activate == "relu":
            return tf.nn.relu(conv_biased, name = '{0}_transpose_conv'.format(idx))
        elif activate == "leaky_relu":
            return tf.nn.leaky_relu(conv_biased, name = '{0}_transpose_conv'.format(idx))
        elif activate == "sigmoid":
            return tf.nn.sigmoid(conv_biased, name ='sigmoid') 
        else:
            return conv_biased
    

def fc_layer(inputs, hiddens, idx, flat=False, activate="relu",weight_init=0.01,initializer=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope('{0}_fc'.format(idx)) as scope:
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_processed = tf.reshape(inputs, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weights = _variable_with_weight_decay('weights', shape = [dim, hiddens], stddev = weight_init,
                                              wd = weight_decay)
        biases = _variable_on_cpu('biases', [hiddens],initializer)
        if activate == "linear":
            return tf.add(tf.matmul(inputs_processed, weights), biases, name = str(idx) + '_fc')
        elif activate == "sigmoid":
            return tf.nn.sigmoid(tf.add(tf.matmul(inputs_processed, weights), biases, name = str(idx) + '_fc'))
        elif activate == "softmax":
            return tf.nn.softmax(tf.add(tf.matmul(inputs_processed, weights), biases, name = str(idx) + '_fc'))
        elif activate == "relu":
            return tf.nn.relu(tf.add(tf.matmul(inputs_processed, weights), biases, name = str(idx) + '_fc'))
        elif activate == "leaky_relu":
            return tf.nn.leaky_relu(tf.add(tf.matmul(inputs_processed, weights), biases, name = str(idx) + '_fc'))        
        else:
            ip = tf.add(tf.matmul(inputs_processed, weights), biases)
            return tf.nn.elu(ip, name = str(idx) + '_fc')
        
def bn_layers(inputs,idx,is_training=True,epsilon=1e-3,decay=0.99,reuse=None):
    with tf.variable_scope('{0}_bn'.format(idx)) as scope:
        #Calculate batch mean and variance
        shape = inputs.get_shape().as_list()
        scale = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=is_training)
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=is_training)
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
        
        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean,train_var]):
                 return tf.nn.batch_normalization(inputs,batch_mean,batch_var,beta,scale,epsilon)
        else:
             return tf.nn.batch_normalization(inputs,pop_mean,pop_var,beta,scale,epsilon)

def bn_layers_wrapper(inputs, is_training):
    pass
   