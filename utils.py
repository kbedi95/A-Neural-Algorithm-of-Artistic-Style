import tensorflow as tf
import urllib.request
import numpy as np
from skimage.transform import resize
import os
import functools


def preprocess(img0to255):

    # row and height of new cropped image
    num_rows, num_cols = img0to255.shape[0], img0to255.shape[1]

    # Scale the image down from [0 255] range to [0 1]
    img0to1 = img0to255 / 255.0

    # New cropped image will have be square with the size of each dimension
    # being the smaller of its num_rows and num_cols
    crop_size = min(num_rows, num_cols)

    # extra pixels along each dimension of the original image (one of these will be 0)
    extra_y = num_rows - crop_size
    extra_x = num_cols - crop_size

    # starting location along each dimension of the original image for
    # the cropped image (one of these will be 0)
    # this will have the larger dimension cropped evenly along both ends
    y_start = extra_y // 2
    x_start = extra_x // 2

    # Crop image using calculated starting position along each dimension and crop_size
    # for number of pixels along each dimension
    cropped_img = img0to1[y_start : y_start + crop_size, x_start : x_start + crop_size]

    # resize the cropped image to (224, 224)
    return resize(cropped_img, (224, 224), preserve_range=True).astype(np.float32)


def deprocess(img0to1):

    # Scale image range to [0 255]
    img0to255 = np.clip(img0to1 * 255, 0, 255)

    # Convert scaled values to integers
    return img0to255.astype(np.uint8)


def get_vgg_model():

    if not os.path.exists("vgg16.tfmodel"):
        print("Downloading vgg16 tensorflow model...")

        # Special thanks to cadl for providing the vgg16 tensorflow model in an accessible location
        # Download the model to the local directory
        urllib.request.urlretrieve("https://s3.amazonaws.com/cadl/models/vgg16.tfmodel", "vgg16.tfmodel")

    # Open the model for reading in binary mode
    vgg16_file = open("vgg16.tfmodel", mode='rb')

    # Load the vgg16 model into the vgg16_graph_def variable
    vgg16_graph_def = tf.GraphDef()
    vgg16_graph_def.ParseFromString(vgg16_file.read())

    return vgg16_graph_def


# Runs a symbolic_function using session with the dropout layer of the network
# consisting of all 1 probabilities (no dropout)
def run_without_dropout(session, symbolic_function, feed_dict={}):

    # Set dropout layers in the model to all 1s so no dropout occurs
    no_dropout = {'vgg/dropout_1/random_uniform:0': [[1.0] * 4096],
                  'vgg/dropout/random_uniform:0': [[1.0] * 4096]}

    # Add additional parameters to run with given session
    no_dropout.update(feed_dict)

    return session.run(symbolic_function, feed_dict=no_dropout)


def get_layer_activations(session, graph, layer_name, input_layer_name, input):

    # Get graph tensor corresponding to given layer name
    tensor = graph.get_tensor_by_name(layer_name)

    # Run without dropout to make result more deterministic
    return run_without_dropout(session, tensor, feed_dict={
        input_layer_name: input
    })


# Returns the total number of values in a tensor
def tensor_size(tensor):

    size = 1
    tensor_shape = tensor.get_shape().as_list()

    for i in range(len(tensor_shape)):
        size *= tensor_shape[i]

    return size


def symbolic_gram(style_tensor):

    # Symbolic output activations for each filter convolution
    # Shape is: activations x channels
    channel_activations = tf.reshape(style_tensor,
                                     [-1, style_tensor.get_shape().as_list()[-1]])

    # Symbolic inner product between all feature activations scaled by total number of activations
    # Shape is: (channels x activations) x (activations x channels) = channels x channels
    return tf.matmul(tf.transpose(channel_activations), channel_activations) / tensor_size(channel_activations)


def gram_matrix(style_activation):

    # Output activations for each filter convolution
    # Shape is: activations x channels
    channel_activations = np.reshape(style_activation, [-1, style_activation.shape[-1]])

    # Inner Product between all feature activations scaled by total number of activations
    # Shape is: (channels x activations) x (activations x channels) = channels x channels
    return np.matmul(channel_activations.T, channel_activations) / channel_activations.size


# Special thanks to Danijar Hafner for this approach and piece of code (lazy_property):
# https://danijar.com/structuring-your-tensorflow-models/
def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        # If the class's method has not already been called, call the method and store the result
        # in an attribute of the class with name: attribute ("_cache_" + function._name_)
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))

        # Otherwise just return the attribute of the class corresponding to the result of the desired
        # method
        return getattr(self, attribute)

    return decorator
