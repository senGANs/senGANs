import numpy as np
import tensorflow as tf
import math


# variable
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)


# define 1d conv
def conv1d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = int(input_.get_shape()[0])
    input_heigh = int(input_.get_shape()[-1])
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, input_heigh, output_dim])
        output = tf.nn.conv1d(input_, kernel, stride, padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# define 2d conv
def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# define 2d deconv
def deconv2d(input_, output_dim, kernel_size, strides, padding="SAME", name="deconv2d"):
    input_dim = int(input_.get_shape()[-1])
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    input_batch_size = int(input_.get_shape()[0])
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[input_width, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel,
                                        output_shape=[input_batch_size, input_height * 2, input_width, output_dim],
                                        strides=[1, strides[0], strides[1], 1], padding="SAME")
        return output


# define batchnorm
def batch_norm(input_, name="batch_norm", reuse=True):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


# define lrelu
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# define generator
def generator(input_, gf_dim=32, reuse=False, name="generator"):
    input_dim = int(input_.get_shape()[-1])
    # dropout_rate = 0.2
    #input_ [1, 10, 10, 1]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        g1 = batch_norm(
            deconv2d(input_=input_, output_dim=gf_dim, kernel_size=3, strides=[2, 1], padding='SAME', name='g1'),
            name='g1_bn')

        g2 = batch_norm(
            deconv2d(input_=lrelu(g1), output_dim=gf_dim * 2, kernel_size=3, strides=[2, 1], padding='SAME',
                     name='g2'), name='g2_bn')
        g3 = batch_norm(
            deconv2d(input_=lrelu(g2), output_dim=gf_dim * 4, kernel_size=3, strides=[2, 1], padding='SAME',
                     name='g3'), name='g3_bn')
        g4 = batch_norm(
            deconv2d(input_=lrelu(g3), output_dim=gf_dim * 8, kernel_size=3, strides=[2, 1], padding='SAME',
                     name='g4'), name='g4_bn')
        g5 = batch_norm(
            deconv2d(input_=lrelu(g4), output_dim=gf_dim * 8, kernel_size=3, strides=[2, 1], padding='SAME',
                     name='g5'), name='g5_bn')

        g6 = batch_norm(
            conv1d(input_=lrelu(tf.reshape(g5, [1, 3200, gf_dim * 8])), output_dim=gf_dim * 8, kernel_size=3, stride=2,
                   padding='SAME',
                   name='g6'), name='g6_bn')  # dim 1

        g7 = batch_norm(conv1d(input_=lrelu(g6), output_dim=gf_dim * 8, kernel_size=3, stride=2, padding='SAME', name='g7'),
                        name='g7_bn')
        g8 = batch_norm(conv1d(input_=lrelu(g7), output_dim=gf_dim * 4, kernel_size=3, stride=2, padding='SAME', name='g8'),
                        name='g8_bn')
        g9 = batch_norm(conv1d(input_=lrelu(g8), output_dim=gf_dim * 2, kernel_size=3, stride=2, padding='SAME', name='g9'),
                        name='g9_bn')
        g10 = batch_norm(conv1d(input_=lrelu(g9), output_dim=1, kernel_size=3, stride=2, padding='SAME', name='g10'),
                         name='g10_bn')

        return tf.nn.sigmoid(g10)


# define discriminator
def discriminator(targets, df_dim=64, reuse=False, name="discriminator"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # targets [1, 10, 10, 1]
        h0 = lrelu(conv2d(input_=targets, output_dim=df_dim, kernel_size=2, stride=2, name='d_h0_conv'))

        h1 = lrelu(batch_norm(conv2d(input_=h0, output_dim=df_dim * 2, kernel_size=2, stride=2, name='d_h1_conv'),
                              name='d_bn1'))

        h2 = lrelu(batch_norm(conv2d(input_=h1, output_dim=df_dim * 2, kernel_size=2, stride=1, name='d_h2_conv'),
                              name='d_bn2'))

        h3 = lrelu(batch_norm(conv2d(input_=h2, output_dim=df_dim * 4, kernel_size=2, stride=1, name='d_h3_conv'),
                              name='d_bn3'))

        output = conv2d(input_=h3, output_dim=1, kernel_size=2, stride=1, name='d_h4_conv')
        dis_out = tf.sigmoid(output)
        return dis_out
