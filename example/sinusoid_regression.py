import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from maml_wrapper.maml_wrapper import MAMLWrapper


def model(input, n_hidden=100):
    hidden = slim.fully_connected(input, n_hidden, activation_fn=tf.nn.sigmoid)
    hidden = slim.fully_connected(hidden, n_hidden, activation_fn=tf.nn.sigmoid)
    return slim.fully_connected(hidden, 1, activation_fn=None)


def main():
    meta_batch_size = 1000

    with tf.name_scope('inputs'):
        x_update_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        x_meta_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        y_update_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        y_meta_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    # ______________________________________________________________________________________________
    # trainable update learning rate
    with tf.name_scope('update_learning_rate'):
        lr = tf.Variable(1., dtype=tf.float32)
        lr = tf.nn.sigmoid(lr) * 1e-2

    n_update_steps = 5

    def mse(y_target, y_prediction):
        return tf.reduce_mean((y_target - y_prediction)**2)

    maml_wrapper = MAMLWrapper(n_update_steps, lr, stop_gradients=True)
    y_meta = maml_wrapper(model, mse, x_update_, y_update_, x_meta_)

    with tf.name_scope('meta_loss'):
        meta_loss = tf.reduce_mean((y_meta - y_meta_)**2)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(1e-3)
        gradient_accumulate, optimize_step = maml_wrapper.meta_optimize(optimizer, meta_loss, meta_batch_size)

    def gen_data(phase):
        x_batch_update = np.random.uniform(-1, 1, size=(n_batch,))
        x_batch_meta = np.random.uniform(-1, 1, size=(n_batch,))
        y_batch_update = np.sin(x_batch_update + phase)
        y_batch_meta = np.sin(x_batch_meta + phase)
        return x_batch_update, y_batch_update, x_batch_meta, y_batch_meta

    board_writer = tf.summary.FileWriter('tensorboard_log', tf.get_default_graph())
    n_batch = 10
    print_every = 10

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for j in range(1000):
            # ______________________________________________________________________________________
            # need to manually batch tasks
            for i in range(meta_batch_size):
                phase = np.random.uniform(0, np.pi)
                x_batch_update, y_batch_update, x_batch_meta, y_batch_meta = gen_data(phase)

                feed_dict = {
                    x_update_: x_batch_update[:, None],
                    x_meta_: x_batch_meta[:, None],
                    y_update_: y_batch_update[:, None],
                    y_meta_: y_batch_meta[:, None]
                }

                session.run(gradient_accumulate, feed_dict=feed_dict)
            session.run(optimize_step)

            if j % print_every == 0:
                uu, mm = [], []
                for _ in range(100):
                    phase = np.random.uniform(0, np.pi)
                    x_batch_update, y_batch_update, x_batch_meta, y_batch_meta = gen_data(phase)

                    feed_dict = {
                        x_update_: x_batch_update[:, None],
                        x_meta_: x_batch_meta[:, None],
                        y_update_: y_batch_update[:, None],
                        y_meta_: y_batch_meta[:, None]
                    }
                    u, m = session.run([maml_wrapper.update_losses[0], meta_loss], feed_dict=feed_dict)
                    uu.append(u)
                    mm.append(m)
                print('{:6.4f} | {:6.4f}'.format(np.mean(uu), np.mean(mm)))


if __name__ == '__main__':
    main()