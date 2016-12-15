import tensorflow as tf
import numpy as np

def fast_weights_encoding(batch, input_size, state, FLAGS, DO_SHARE):
    """
    :param FLAGS:
        input_dim = 1  # 9
        slow_init = 0.05 # Slow weights initialization scaling [default = 0.05] (See Hinton's video @ 21:20)
        num_hidden_units = 10  # Hidden units [default = 100] [50,100,200]
        decay_lambda = 0.9  # decay lambda value [default = 0.9] [0.9, 0.95]
        rate_eta = 0.5  # learning rate(eta) [default = 0.5]
        S = 3  # inner loops where h(t+1) is transformed into h_S(t+1) [default = 3]
    :param DO_SHARE:
        share variable weights
    :return:
        fast_weights of shape (batch_size, hidden_units)
    """

    input_dim = FLAGS.input_dim
    slow_init = FLAGS.slow_init
    num_hidden_units = FLAGS.num_hidden_units
    decay_lambda = FLAGS.decay_lambda
    rate_eta = FLAGS.rate_eta
    S = FLAGS.S
    T = FLAGS.T
    batch_size = FLAGS.batch_size

    # Fast weight graph variables
    x_t = tf.placeholder(tf.float32,
                        shape=[None, input_size], name='input_x_t')
    a_lambda = tf.placeholder(tf.float32, [],  # need [] for tf.scalar_mul
                                name="decay_rate")
    e_rate = tf.placeholder(tf.float32, [], name="learning_rate")

    a_lambda = decay_lambda # decay rate of weights
    e_rate = rate_eta # learning rate for weights

    x_t = batch

    with tf.variable_scope(state, reuse=DO_SHARE):
        # input weights (proper initialization)
        W_x = tf.Variable(tf.random_uniform(
            [input_size, num_hidden_units],
            -np.sqrt(2.0 / T), np.sqrt(2.0 / T)),
            dtype=tf.float32, name = "W_x")
        b_x = tf.Variable(
            tf.zeros([num_hidden_units]),
            dtype=tf.float32, name = "b_x")

        # hidden "slow" weights [Supplementary Material A-A.1]
        W_h = tf.Variable(
            initial_value=slow_init * np.identity(num_hidden_units),
            dtype=tf.float32, name = "W_h")

        # scale and shift for layernorm
        gain = tf.Variable(tf.ones(
            [num_hidden_units]),
            dtype=tf.float32, name = "gain")
        bias = tf.Variable(tf.zeros(
            [num_hidden_units]),
            dtype=tf.float32, name = "bias")

    # fast weights and hidden state initialization
    A_fast = tf.zeros(
        #[1, batch_size, batch_size],
        [batch_size, num_hidden_units, num_hidden_units],
        dtype=tf.float32, name = "A_fast")
    h_fast = tf.zeros(
        #[1, batch_size],
        [batch_size, num_hidden_units],
        dtype=tf.float32, name = "h_fast")

    # NOTE:inputs are batch-major
    # Process batch by time-major
    for t in range(0, input_dim):
        h_fast = tf.nn.relu((tf.matmul(x_t, W_x) + b_x) +
                            (tf.matmul(h_fast, W_h)))

        # Forward weight and layer normalization
        # Reshape h to use with a
        h_s = tf.reshape(h_fast,
                         [batch_size, 1, num_hidden_units])

        # Create the fixed A for this time step
        A_fast = tf.add(tf.scalar_mul(a_lambda, A_fast),
                        tf.scalar_mul(e_rate, tf.batch_matmul(tf.transpose(
                            h_s, [0, 2, 1]), h_s)))

        # Loop for S steps
        for _ in range(S):
            h_s = tf.reshape(
                tf.matmul(x_t, W_x) + b_x,
                tf.shape(h_s)) + tf.reshape(
                tf.matmul(h_fast, W_h), tf.shape(h_s)) + tf.batch_matmul(h_s, A_fast)

            # Apply layernorm
            mu = tf.reduce_mean(h_s, reduction_indices=0)  # each sample
            sigma = tf.sqrt(tf.reduce_mean(tf.square(h_s - mu),
                                           reduction_indices=0))
            h_s = tf.div(tf.mul(gain, (h_s - mu)), sigma) + bias

            # Apply nonlinearity
            h_s = tf.nn.relu(h_s)

        # Reshape h_s into h
        h_fast = tf.reshape(h_s,
                            #[batch_size, batch_size])
                            [batch_size, num_hidden_units])

        return h_fast
