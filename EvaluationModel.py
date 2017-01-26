import numpy as np
from utils import lazy_property
import utils
import tensorflow as tf
import yaml
import pylab as plt
config = yaml.safe_load(open("config.yaml"))


class EvaluationModel(object):

    def __init__(self, content_path, style_path):

        # 4D representations of the given content and style images
        self.content_image = utils.preprocess(plt.imread(content_path))[np.newaxis]
        self.style_image = utils.preprocess(plt.imread(style_path))[np.newaxis]

        # The session and graph used for evaluating the content and style of the
        # given content and style images
        self.evaluation_g = tf.Graph()
        self.evaluation_sess = tf.Session(graph=self.evaluation_g)

        # The outputs (:0) of the intermediate layers of the VGG16 model used to represent the
        # content and style of an input to the model
        self.content_layer = config["content_layer"]
        self.style_layers = config["style_layers"]

        with self.evaluation_g.as_default():

            # Import the VGG16 ImageNet predictor model graph into the evaluation_g member variable
            tf.import_graph_def(utils.get_vgg_model(), name="vgg")

            # The input to the VGG16 predictor model is the output (:0) of the first operation of the graph
            self.input_tensor = [op.name for op in self.evaluation_g.get_operations()][0] + ":0"

    @lazy_property
    def content_activations(self):

        # Get the activations of the given content image at the given content layer
        return utils.get_layer_activations(session=self.evaluation_sess, graph=self.evaluation_g,
                                           layer_name=self.content_layer, input_layer_name=self.input_tensor,
                                           input=self.content_image)

    @lazy_property
    def style_gram_matrices(self):

        gram_matrices = []

        for style_layer in config["style_layers"]:

            # Get the activations of the given style image at the given style layer
            style_activation = utils.get_layer_activations(session=self.evaluation_sess, graph=self.evaluation_g,
                                                           layer_name=style_layer, input_layer_name=self.input_tensor,
                                                           input=self.style_image)

            # Construct the gram matrix from the activations and save the matrix to the gram_matrices array
            gram_matrices.append(utils.gram_matrix(style_activation).astype(np.float32))

        return gram_matrices
