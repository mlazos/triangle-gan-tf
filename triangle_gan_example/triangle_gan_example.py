import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.flags.DEFINE_integer(
    "batch_size",
    64,
    "The number of samples to train the model on each iteration.")

tf.flags.DEFINE_integer(
    "num_generator_layers",
    3,
    "The number of layers in the generator networks.")

tf.flags.DEFINE_integer(
    "num_discriminator_layers",
    4,
    "The number of layers in the discriminator networks.")

tf.flags.DEFINE_integer(
    "num_generator_units",
    32,
    "The number of units in each layer of the generators.")

tf.flags.DEFINE_integer(
    "num_discriminator_units",
    32,
    "The number of units in each layer of the discriminators.")

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

hidden_activation = tf.nn.tanh
num_discriminator_layers = tf.flags.FLAGS.num_discriminator_layers
num_discriminator_units = tf.flags.FLAGS.num_discriminator_units
num_generator_layers = tf.flags.FLAGS.num_generator_layers
num_generator_units = tf.flags.FLAGS.num_generator_units

def generator(input):
    """
    Creates a generator network with the given input.

    Arguments:
        input: The input to the network.
    
    Returns:
        A sample from the generator distribution.
    """
    return feedforward(input, 2, num_generator_units, 1, num_generator_layers, hidden_activation)

def discriminator(input):
    """
    Creates a discriminator network with the given input.

    Arguments:
        input: The input to the network.

    Returns:
        A score of whether the sample is from the desired distribution or not.
    """
    return feedforward(input, 2, num_discriminator_units, 1, num_discriminator_layers, hidden_activation, tf.nn.sigmoid)

def generate_samples(num_samples_per_component):
    """
    Generates num_samples for each component of a uniform mixture of four 2D gaussians.

    Arguments:
        num_samples_per_component: The number of samples to generate form each component of the uniform gaussian mixture.

    Returns:
        A numpy array of shape [4 * num_samples_per_component, 2] where each row is a single sample.
    """
    return np.concatenate(
        (np.random.multivariate_normal([0,1.5], [[3,0],[0,0.025]], num_samples_per_component),
        np.random.multivariate_normal([-1.5,0], [[0.025,0],[0,3]], num_samples_per_component),
        np.random.multivariate_normal([1.5,0], [[0.025,0],[0,3]], num_samples_per_component),
        np.random.multivariate_normal([0,-1.5], [[3,0],[0,0.025]], num_samples_per_component)))

joint_samples = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="joint_samples")
x_samples = tf.placeholder(dtype=tf.float32, shape=[None], name="x_samples")
x_samples_e = tf.expand_dims(x_samples, 1)
y_samples = tf.placeholder(dtype=tf.float32, shape=[None], name="y_samples")
y_samples_e = tf.expand_dims(y_samples, 1)
z_samples = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="z_samples")
x_z_samples = tf.concat((x_samples_e, z_samples), 1)
z_y_samples = tf.concat((z_samples, y_samples_e), 1)

batch_size = tf.shape(x_samples)[0]
tf.assert_equal(tf.shape(x_samples)[0], tf.shape(y_samples)[0])
tf.assert_equal(tf.shape(x_samples)[0], tf.shape(z_samples)[0])
tf.assert_equal(tf.shape(x_samples)[0], tf.shape(joint_samples)[0])

with tf.variable_scope("generator_xy") as g_xy_scope:
    x_given_y = generator(z_y_samples)

with tf.variable_scope("generator_yx") as g_yx_scope:
    y_given_x = generator(x_z_samples)

fake_x_pairs = tf.concat((x_given_y, y_samples_e), 1)
fake_y_pairs = tf.concat((x_samples_e, y_given_x), 1)

test_samples = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="test")

with tf.variable_scope("disc_real_or_fake") as d_rf_scope:
    fake_x_scores_drf = discriminator(fake_x_pairs)
    d_rf_scope.reuse_variables()
    real_scores_drf = discriminator(joint_samples)
    fake_y_scores_drf = discriminator(fake_y_pairs)
    decision_real_or_fake = discriminator(test_samples)

with tf.variable_scope("disc_fake_x_or_fake_y") as d_fxfy_scope:
    fake_x_scores_dfxfy = discriminator(fake_x_pairs)
    d_fxfy_scope.reuse_variables()
    fake_y_scores_dfxfy = discriminator(fake_y_pairs)
    decision_fake_x_or_fake_y = discriminator(test_samples)

ONES = tf.ones([batch_size, 1])
ZEROS = tf.zeros([batch_size, 1])

loss_real_fake_disc = tf.losses.log_loss(ZEROS, fake_x_scores_drf) \
    + tf.losses.log_loss(ONES, real_scores_drf) \
    + tf.losses.log_loss(ZEROS, fake_y_scores_drf)

loss_fake_x_fake_y_disc = tf.losses.log_loss(ONES, fake_y_scores_dfxfy) \
    + tf.losses.log_loss(ZEROS, fake_x_scores_dfxfy)

loss_discriminators = loss_real_fake_disc + loss_fake_x_fake_y_disc

loss_gen_x_y = tf.losses.log_loss(ONES, fake_x_scores_drf) \
    + tf.losses.log_loss(ONES, fake_x_scores_dfxfy)

loss_gen_y_x = tf.losses.log_loss(ONES, fake_y_scores_drf) \
    + tf.losses.log_loss(ZEROS, fake_y_scores_dfxfy)

loss_generators = loss_gen_x_y + loss_gen_y_x

generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator_xy") \
    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator_yx")

discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="disc_real_or_fake") \
    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="disc_fake_x_or_fake_y")

train_step_discriminators = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(loss_discriminators, var_list=discriminator_vars)
train_step_generators = tf.train.AdamOptimizer(learning_rate=0.0025).minimize(loss_generators, var_list=generator_vars)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(0, 1000):
    joint_input = generate_samples(125)

    np.random.shuffle(joint_input)

    temp_input = generate_samples(125)
    
    np.random.shuffle(joint_input)
    np.random.shuffle(temp_input)
    x_input = temp_input[:, 0]
    np.random.shuffle(temp_input)
    y_input = temp_input[:, 1]
    z_input = np.random.multivariate_normal([0], [[10.0]], 4 * 125)

    for _ in range(0, 10):
        train_step_discriminators.run({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})

    train_step_generators.run({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})

    if (i % 100 == 0):
        loss_g = loss_generators.eval({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})
        loss_d = loss_discriminators.eval({x_samples:x_input, joint_samples:joint_input, y_samples:y_input, z_samples: z_input})
        print("iteration:" + str(i) + ", generator_loss=" + str(loss_g) + " disc_loss =" + str(loss_d))

final_samples = generate_samples(125)

final_x = final_samples[:, 0]
final_y = final_samples[:, 1]
final_z = np.random.multivariate_normal([0], [[10.0]], 500)

final_x_given_y = x_given_y.eval({y_samples: final_y, z_samples: final_z})
final_y_given_x = y_given_x.eval({x_samples: final_x, z_samples: final_z})

# Print the final distributions
plt.plot(final_x, final_y_given_x, "o")
plt.axis('equal')
plt.show()

plt.plot(final_x_given_y, final_y, "o")
plt.axis('equal')
plt.show()

plt.plot(final_x, final_y, "o")
plt.axis('equal')
plt.show()

# Print the decision boundary heat maps
test_samples_v = np.transpose(np.reshape(np.mgrid[-5:5:0.05, -5:5:0.05], [2, -1]))
real_fake_boundary = np.reshape(decision_real_or_fake.eval({test_samples:test_samples_v}), [200, 200])
fake_x_fake_y_boundary = np.reshape(decision_fake_x_or_fake_y.eval({test_samples:test_samples_v}), [200, 200])
plt.imshow(real_fake_boundary, cmap='hot', interpolation='nearest')
plt.show()
plt.imshow(fake_x_fake_y_boundary, cmap='hot', interpolation='nearest')
plt.show()