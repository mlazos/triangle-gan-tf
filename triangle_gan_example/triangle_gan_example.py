import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.flags.DEFINE_integer(
    "batch_size",
    64,
    "The number of samples to train the model on each iteration.")

X = np.random.normal(0, 1.0, 1000)
Y = np.random.normal(0, 1.0, 1000)

def feedforward(
        inputs,
        hidden_dim,
        output_dim,
        num_hidden_layers,
        hidden_activation=None,
        output_activation=None):
    """
    Creates a dense feedforward network with num_hidden_layers layers where each layer
    has hidden_dim number of units except for the last layer which has output_dim number of units.

    Arguments:
        inputs: Tensor input.
        hidden_dim: The number of units in each hidden layer.
        output_dim: The number of units in the output layer.
        num_hidden_layers: The number of hidden layers.
        hidden_activation: The activation function of hidden layers.
            Set it to None to use a linear activation.
        output_activation: The activation function of the output layer.
            Set it to None to use a linear activation.

    Returns:
        Output tensor.
    """
    prev_output = inputs
    for _ in range(0, num_hidden_layers):
        prev_output = tf.layers.dense(prev_output, hidden_dim, activation=hidden_activation)
    return tf.layers.dense(prev_output, output_dim, activation=output_activation)

joint_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 2])
x_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 1])
y_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 1])
x_z_samples = tf.concat((x_samples, tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 2])), 1)
z_y_samples = tf.concat((tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 2]), y_samples), 1)

with tf.variable_scope("generator_xy"):
    x_given_y = feedforward(z_y_samples, 512, 1, 4, tf.nn.leaky_relu)

with tf.variable_scope("generator_yx"):
    y_given_x = feedforward(x_z_samples, 512, 1, 4, tf.nn.leaky_relu)

with tf.variable_scope("discriminator_real_fake"):
    disc_real_fake = feedforward(x_given_y, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)

with tf.variable_scope("discriminator_fake_x_fake_y"):
    disc_fake_x_fake_y = feedforward(y_given_x, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)

#print(X)
#plt.plot(X, Y, 'x')
#plt.axis('equal')
#plt.show()
