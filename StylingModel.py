import numpy as np
import tensorflow as tf
import utils
from utils import lazy_property
from PIL import Image
from EvaluationModel import EvaluationModel
import yaml
config = yaml.safe_load(open("config.yaml"))


class StylingModel(object):

    def __init__(self, content_path, style_path):

        # Evaluation model is used to evaluate the content and style information
        # of the given content and style images
        self.evaluation_model = EvaluationModel(content_path=content_path, style_path=style_path)

        # The session and graph used for styling a given content image
        self.styling_g = tf.Graph()
        self.styling_sess = tf.Session(graph=self.styling_g)

        # Import the VGG16 model into the Style_Model's graph with the graph's input image
        # as a a variable to be optimized
        with self.styling_g.as_default():

            self.styled_image = tf.Variable(self.evaluation_model.content_image)
            tf.import_graph_def(
                utils.get_vgg_model(),
                name="vgg",
                input_map={"images:0": self.styled_image})

    @lazy_property
    def content_loss(self):

        with self.styling_sess.as_default():

            # Get the activation values for the generated and content image at the content layer
            gen_img_activations = self.styling_g.get_tensor_by_name(self.evaluation_model.content_layer)
            content_img_activations = self.evaluation_model.content_activations

            # Difference between the generated image and content image's activations at the content layer
            content_difference = gen_img_activations - content_img_activations

            # Loss is the L2 loss of the difference between activations scaled by the number of values in the
            # output of the content layer
            return tf.nn.l2_loss(content_difference / self.evaluation_model.content_activations.size)

    @lazy_property
    def style_loss(self):

        with self.styling_sess.as_default():

            total_loss = np.float32(0.0)

            for i in range(len(config["style_layers"])):
                # Get the activation values for the generated image at the current style layer
                gen_img_activations = self.styling_g.get_tensor_by_name(config["style_layers"][i])

                # Construct symbolic gram matrix of generated image for current style layer
                gen_img_gram = utils.symbolic_gram(gen_img_activations)

                # Get gram matrix of style image for current style layer
                style_img_gram = self.evaluation_model.style_gram_matrices[i]

                # Difference between style image and generated image's gram matrices for current style layer
                gram_difference = gen_img_gram - style_img_gram

                # The loss for current style layer is the L2 loss of the scaled difference between gram matrices
                # (difference is scaled by number of values in the gram matrix)
                layer_loss = tf.nn.l2_loss(gram_difference / np.float32(style_img_gram.size))

                total_loss += layer_loss

            return total_loss

    @lazy_property
    def tv_loss(self):

        with self.styling_g.as_default():

            # height and width of 4D square input image
            height = self.styled_image.get_shape().as_list()[1]
            width = height

            # Input image reduced to exclude last row and column (to preserve square)
            reduced = self.styled_image[:, :height - 1, :width - 1, :]

            # Input image reduced to exclude last row and shift columns left
            left_shift_reduced = self.styled_image[:, :height - 1, 1:, :]

            # Input image reduced to exclude last columns and shift columns up
            up_shift_reduced = self.styled_image[:, 1:, :width - 1, :]

            # difference between horizontally neighboring pixels
            horizontal_diff = tf.square(reduced - left_shift_reduced)

            # difference between vertically neighboring pixels
            vertical_diff = tf.square(reduced - up_shift_reduced)

            # Sum the neighboring differences for each pixel and raise to 1.25 power
            # Sum over all of these values
            return tf.reduce_sum((horizontal_diff + vertical_diff) ** 1.25)

    def optimize(self, content_weight, style_weight, tv_weight):

        with self.styling_g.as_default():

            # Weight the content loss, style loss, and tv loss of the generated image as required
            loss = content_weight * self.content_loss + \
                   style_weight * self.style_loss + \
                   tv_weight * self.tv_loss

            # Perform gradient descent with the Adam optimization algorithm
            # with a step size of 0.01
            self.optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

    def train(self, iterations, frequency, target_path):

        with self.styling_sess.as_default(), self.styling_g.as_default():

            # Initialize variables in vgg16 model (dropout etc.)
            self.styling_sess.run(tf.initialize_all_variables())

            for it_i in range(iterations):

                # Run optimizer without dropout to make model more deterministic
                utils.run_without_dropout(self.styling_sess, self.optimizer)

                # Print step of gradient descent
                print("%d" % it_i)

                # Save generated image to target path with required frequency
                if it_i % frequency == 0:
                    generated_image4D = utils.run_without_dropout(self.styling_sess, self.styled_image)

                    # Choose the first image in the batch of 3D images: generated_image4D
                    # (there is only one image in the batch), deprocess it, and save the image
                    Image.fromarray(utils.deprocess(generated_image4D[0])).save(target_path + "img-%d.png" % it_i)
