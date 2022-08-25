"""
Functions for bilinear or trilinear resizing.
"""

import tensorflow as tf
from tensorflow_train.utils.data_format import get_batch_channel_image_size, get_image_size


def resize_trilinear(inputs, factors=None, output_size=None, name=None, data_format='channels_first'):
    """
    Trilinearly resizes an input volume to either a given size of a factor.
    :param inputs: 5D tensor.
    :param output_size: Output size.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    num_batches, num_channels, [depth, height, width] = get_batch_channel_image_size(inputs, data_format)
    dtype = inputs.dtype
    name = name or 'upsample'
    with tf.compat.v1.name_scope(name):
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(a=inputs, perm=[0, 2, 3, 4, 1])
        else:
            inputs_channels_last = inputs

        if output_size is None:
            output_depth, output_height, output_width = [int(s * f) for s, f in zip([depth, height, width], factors)]
        else:
            output_depth, output_height, output_width = output_size

        # resize y-z
        squeeze_b_x = tf.reshape(inputs_channels_last, [-1, height, width, num_channels])
        resize_b_x = tf.cast(tf.image.resize(squeeze_b_x, [output_height, output_width], method=tf.image.ResizeMethod.BILINEAR, half_pixel_centers=True), dtype=dtype)
        resume_b_x = tf.reshape(resize_b_x, [num_batches, depth, output_height, output_width, num_channels])

        # resize x
        #   first reorient
        reoriented = tf.transpose(a=resume_b_x, perm=[0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, output_height, depth, num_channels])
        resize_b_z = tf.cast(tf.image.resize(squeeze_b_z, [output_height, output_depth], method=tf.image.ResizeMethod.BILINEAR, half_pixel_centers=True), dtype=dtype)
        resume_b_z = tf.reshape(resize_b_z, [num_batches, output_width, output_height, output_depth, num_channels])

        if data_format == 'channels_first':
            output = tf.transpose(a=resume_b_z, perm=[0, 4, 3, 2, 1])
        else:
            output = tf.transpose(a=resume_b_z, perm=[0, 3, 2, 1, 4])
        return output


def resize_tricubic(inputs, factors=None, output_size=None, name=None, data_format='channels_first'):
    """
    Trilinearly resizes an input volume to either a given size of a factor.
    :param inputs: 5D tensor.
    :param output_size: Output size.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    num_batches, num_channels, [depth, height, width] = get_batch_channel_image_size(inputs, data_format)
    dtype = inputs.dtype
    name = name or 'upsample'
    with tf.compat.v1.name_scope(name):
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(a=inputs, perm=[0, 2, 3, 4, 1])
        else:
            inputs_channels_last = inputs

        if output_size is None:
            output_depth, output_height, output_width = [int(s * f) for s, f in zip([depth, height, width], factors)]
        else:
            output_depth, output_height, output_width = output_size

        # resize y-z
        squeeze_b_x = tf.reshape(inputs_channels_last, [-1, height, width, num_channels])
        resize_b_x = tf.cast(tf.image.resize(squeeze_b_x, [output_height, output_width], method=tf.image.ResizeMethod.BICUBIC, half_pixel_centers=True), dtype=dtype)
        resume_b_x = tf.reshape(resize_b_x, [num_batches, depth, output_height, output_width, num_channels])

        # resize x
        #   first reorient
        reoriented = tf.transpose(a=resume_b_x, perm=[0, 3, 2, 1, 4])
        #   squeeze and 2d resize
        squeeze_b_z = tf.reshape(reoriented, [-1, output_height, depth, num_channels])
        resize_b_z = tf.cast(tf.image.resize(squeeze_b_z, [output_height, output_depth], method=tf.image.ResizeMethod.BICUBIC, half_pixel_centers=True), dtype=dtype)
        resume_b_z = tf.reshape(resize_b_z, [num_batches, output_width, output_height, output_depth, num_channels])

        if data_format == 'channels_first':
            output = tf.transpose(a=resume_b_z, perm=[0, 4, 3, 2, 1])
        else:
            output = tf.transpose(a=resume_b_z, perm=[0, 3, 2, 1, 4])
        return output


def resize_bilinear(inputs, output_size=None, factors=None, name=None, data_format='channels_first'):
    """
    Bilinearly resizes an input volume to either a given size of a factor.
    :param inputs: 4D tensor.
    :param output_size: Output size.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    input_size = get_image_size(inputs, data_format)
    dtype = inputs.dtype
    name = name or 'upsample'
    with tf.compat.v1.name_scope(name):
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(a=inputs, perm=[0, 2, 3, 1])
        else:
            inputs_channels_last = inputs
        if output_size is None:
            output_size = [int(s * f) for s, f in zip(input_size, factors)]
        upsampled_channels_last = tf.cast(tf.image.resize(inputs_channels_last, output_size, half_pixel_centers=True, method=tf.image.ResizeMethod.BILINEAR), dtype=dtype)
        if data_format == 'channels_first':
            outputs = tf.transpose(a=upsampled_channels_last, perm=[0, 3, 1, 2])
        else:
            outputs = upsampled_channels_last
    return outputs


def resize_bicubic(inputs, output_size=None, factors=None, name=None, data_format='channels_first'):
    """
    Bicubicly resizes an input volume to either a given size of a factor.
    :param inputs: 4D tensor.
    :param output_size: Output size.
    :param factors: Scale factors.
    :param name: Name.
    :param data_format: Data format.
    :return: The resized tensor.
    """
    input_size = get_image_size(inputs, data_format)
    dtype = inputs.dtype
    name = name or 'upsample'
    with tf.compat.v1.name_scope(name):
        if data_format == 'channels_first':
            inputs_channels_last = tf.transpose(a=inputs, perm=[0, 2, 3, 1])
        else:
            inputs_channels_last = inputs
        if output_size is None:
            output_size = [int(s * f) for s, f in zip(input_size, factors)]
        upsampled_channels_last = tf.cast(tf.image.resize(inputs_channels_last, output_size, half_pixel_centers=True, method=tf.image.ResizeMethod.BICUBIC), dtype=dtype)
        if data_format == 'channels_first':
            outputs = tf.transpose(a=upsampled_channels_last, perm=[0, 3, 1, 2])
        else:
            outputs = upsampled_channels_last
    return outputs
