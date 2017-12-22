import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.flags.DEFINE_integer(
    "batch_size",
    64,
    "The number of samples to train the model on each iteration.")

def feedforward(
        inputs,
        input_dim,
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
    prev_input_dim = input_dim
    prev_output = inputs
    for i in range(0, num_hidden_layers):
        with tf.variable_scope("dense" + str(i)):
            w_n = tf.get_variable("w_" + str(i), [prev_input_dim, hidden_dim], initializer=tf.initializers.random_normal(0, 1))
            b_n = tf.get_variable("b_" + str(i), [hidden_dim], initializer=tf.initializers.random_normal(0, 1))
            prev_input_dim = hidden_dim
            prev_output = hidden_activation(tf.matmul(prev_output, w_n) + b_n)
    with tf.variable_scope("dense_output"):
        return tf.layers.dense(prev_output, output_dim, activation=output_activation)

joint_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 2], name="joint_samples")
x_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size], name="x_samples")
x_samples_e = tf.expand_dims(x_samples, 1)
y_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size], name="y_samples")
y_samples_e = tf.expand_dims(y_samples, 1)
z_samples = tf.placeholder(dtype=tf.float32, shape=[tf.flags.FLAGS.batch_size, 2], name="z_samples")
x_z_samples = tf.concat((x_samples_e, z_samples), 1)
z_y_samples = tf.concat((z_samples, y_samples_e), 1)

with tf.variable_scope("generator_xy") as g_xy_scope:
    x_given_y = feedforward(z_y_samples, 3, 512, 1, 4, tf.nn.leaky_relu)

with tf.variable_scope("generator_yx") as g_yx_scope:
    y_given_x = feedforward(x_z_samples, 3, 512, 1, 4, tf.nn.leaky_relu)

fake_x_pairs = tf.concat((x_given_y, y_samples_e), 1)
fake_y_pairs = tf.concat((x_samples_e, y_given_x), 1)

with tf.variable_scope("disc_real_or_fake") as d_rf_scope:
    fake_x_scores_drf = feedforward(fake_x_pairs, 2, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)
    d_rf_scope.reuse_variables()
    real_scores_drf = feedforward(joint_samples, 2, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)
    fake_y_scores_drf = feedforward(fake_y_pairs, 2, 512, 1, 4, tf.nn.sigmoid)

with tf.variable_scope("disc_fake_x_or_fake_y") as d_fxfy_scope:
    fake_x_scores_dfxfy = feedforward(fake_x_pairs, 2, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)
    d_fxfy_scope.reuse_variables()
    fake_y_scores_dfxfy = feedforward(fake_y_pairs, 2, 512, 1, 4, tf.nn.leaky_relu, tf.nn.sigmoid)

loss_real_fake_disc = tf.losses.log_loss(tf.zeros([tf.flags.FLAGS.batch_size, 1]), fake_x_scores_drf) \
    + tf.losses.log_loss(tf.ones([tf.flags.FLAGS.batch_size, 1]), real_scores_drf) \
    + tf.losses.log_loss(tf.zeros([tf.flags.FLAGS.batch_size, 1]), fake_y_scores_drf)

loss_fake_x_fake_y_disc = tf.losses.log_loss(tf.ones([tf.flags.FLAGS.batch_size, 1]), fake_y_scores_dfxfy) \
    + tf.losses.log_loss(tf.zeros([tf.flags.FLAGS.batch_size, 1]), fake_x_scores_dfxfy)

loss_discriminators = loss_real_fake_disc + loss_fake_x_fake_y_disc

loss_gen_x_y = tf.losses.log_loss(tf.ones([tf.flags.FLAGS.batch_size, 1]), fake_x_scores_drf) \
    + tf.losses.log_loss(tf.ones([tf.flags.FLAGS.batch_size, 1]), fake_x_scores_dfxfy)

loss_gen_y_x = tf.losses.log_loss(tf.ones([tf.flags.FLAGS.batch_size, 1]), fake_y_scores_drf) \
    + tf.losses.log_loss(tf.zeros([tf.flags.FLAGS.batch_size, 1]), fake_y_scores_dfxfy)

loss_generators = loss_gen_x_y + loss_gen_y_x

generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator_xy") \
    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator_yx")

discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="disc_real_or_fake") \
    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="disc_fake_x_or_fake_y")

train_step_discriminators = tf.train.AdamOptimizer().minimize(loss_discriminators, var_list=discriminator_vars)
train_step_generators = tf.train.AdamOptimizer().minimize(loss_generators, var_list=generator_vars)

mean2 = [0,1.5]
mean4 = [-1.5,0]
mean6 = [1.5,0]
mean8 = [0,-1.5]
cov2 = [[3,0],[0,0.025]]
cov4 = [[0.025,0],[0,3]]
cov6 = [[0.025,0],[0,3]]
cov8 = [[3,0],[0,0.025]]

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(0, 10):
    joint_input = np.concatenate(
        (np.random.multivariate_normal(mean2, cov2, 50),
        np.random.multivariate_normal(mean4, cov4, 50),
        np.random.multivariate_normal(mean6, cov6, 50),
        np.random.multivariate_normal(mean8, cov8, 50)))

    np.random.shuffle(joint_input)

    temp_input = np.concatenate(
        (np.random.multivariate_normal(mean2, cov2, 50),
        np.random.multivariate_normal(mean4, cov4, 50),
        np.random.multivariate_normal(mean6, cov6, 50),
        np.random.multivariate_normal(mean8, cov8, 50)))
    
    np.random.shuffle(temp_input)
    x_input = temp_input[:, 0]
    np.random.shuffle(temp_input)
    y_input = temp_input[:, 1]
    z_input = np.random.multivariate_normal([0,0], [[100, 100], [100, 100]], tf.flags.FLAGS.batch_size)

    for _ in range(0, 3):
        loss_discriminators.eval({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})
    
    loss_generators.eval({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})

    print("iteration:" + str(i))

final_samples = np.concatenate(
        (np.random.multivariate_normal(mean2, cov2, 50),
        np.random.multivariate_normal(mean4, cov4, 50),
        np.random.multivariate_normal(mean6, cov6, 50),
        np.random.multivariate_normal(mean8, cov8, 50)))

final_x = final_samples[:, 0]
final_y = final_samples[:, 1]
final_z = np.random.multivariate_normal([0, 0], [[100, 100], [100, 100]], tf.flags.FLAGS.batch_size)

final_x_given_y = x_given_y.eval({y_samples: final_y, z_samples: final_z})
final_y_given_x = y_given_x.eval({x_samples: final_x, z_samples: final_z})

print(final_x_given_y)
plt.plot(final_x, np.squeeze(final_y_given_x), "o")
plt.axis('equal')
plt.show()

plt.plot(np.squeeze(final_x_given_y), final_y, "o")
plt.axis('equal')
plt.show()
