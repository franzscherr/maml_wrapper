from francis import *
import tensorflow.contrib.graph_editor as ge


from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_float('meta_lr', 1e-3, 'Meta update learning rate')
flags.DEFINE_float('update_lr', 2e-3, 'Inner update learning rate')
flags.DEFINE_integer('n_update_steps', 5, 'Number of inner updates')
flags.DEFINE_integer('meta_batch_size', 50, 'Different tasks to consider at same time')

logger = get_console_logger('maml')


def model(input, n_hidden=100):
    hidden = slim.fully_connected(input, n_hidden, activation_fn=tf.nn.sigmoid)
    hidden = slim.fully_connected(hidden, n_hidden, activation_fn=tf.nn.sigmoid)
    return slim.fully_connected(hidden, 1, activation_fn=None)


def main():
    with tf.name_scope('inputs'):
        x_update_ = tf.placeholder(dtype=tf.float32, shape=(None, None, 1))
        x_meta_ = tf.placeholder(dtype=tf.float32, shape=(None, None, 1))
        y_update_ = tf.placeholder(dtype=tf.float32, shape=(None, None, 1))
        y_meta_ = tf.placeholder(dtype=tf.float32, shape=(None, None, 1))

    updated_parameters = list()
    update_losses = list()

    # ______________________________________________________________________________________________
    # trainable update learning rate
    lr = tf.Variable(1., dtype=tf.float32)
    lr = tf.nn.sigmoid(lr) * 1e-2

    non_model_trainable_variables = tf.trainable_variables()
    model_variables = []

    x_update_tasks = tf.unstack(x_update_, num=FLAGS.meta_batch_size)
    y_update_tasks = tf.unstack(y_update_, num=FLAGS.meta_batch_size)

    for j in range(FLAGS.meta_batch_size):
        updated_parameters.append([])
        update_losses.append([])
        for i in range(FLAGS.n_update_steps):
            with tf.name_scope('update_{}_task_{}'.format(i, j)):
                with tf.variable_scope('model', reuse=(i != 0 or j != 0)):
                    y_update = model(x_update_tasks[j])
                if i == 0 and j == 0:
                    # __________________________________________________________________________________
                    # determine the models variables
                    model_variables = \
                        list(item for item in tf.trainable_variables() if item not in non_model_trainable_variables)
                if i == 0:
                    updated_parameters[j].append(model_variables)

                update_loss = tf.reduce_mean((y_update - y_update_tasks[j])**2)
                # __________________________________________________________________________________
                # ATTENTION
                # need to differentiate wrt the original parameters because in the graph the updated steps
                # are until now only connected to the original parameters
                grads = tf.gradients(update_loss, model_variables)
                updated_parameters[j].append(list(v - lr * g for v, g in zip(updated_parameters[j][-1], grads)))
                update_losses[j].append(update_loss)
    logger.info('Graph built')

    y_task_meta = []
    for j in range(FLAGS.meta_batch_size):
        with tf.name_scope('meta_task_{}'.format(j)):
            with tf.variable_scope('model', reuse=True):
                y_meta = model(x_meta_)
                y_task_meta.append(y_meta)

    with tf.name_scope('meta_loss'):
        meta_loss = tf.reduce_mean((tf.stack(y_task_meta) - y_meta_)**2)

    # ______________________________________________________________________________________________
    # connect updated parameters to each update step
    for i in range(1, FLAGS.n_update_steps):
        for j in range(FLAGS.meta_batch_size):
            pv = ge.sgv(*updated_parameters[j][i])
            s = ge.sgv_scope('update_{}_task_{}'.format(i, j), tf.get_default_graph())
            ind = list(s.input_index(tf.convert_to_tensor(a)) for a in model_variables)
            s = s.remap_inputs(ind)
            ge.connect(pv, s)

    # ______________________________________________________________________________________________
    # connect final updated parameters
    for j in range(FLAGS.meta_batch_size):
        pv = ge.sgv(*updated_parameters[j][-1])
        s = ge.sgv_scope('meta_task_{}'.format(j), tf.get_default_graph())
        ind = list(s.input_index(tf.convert_to_tensor(a)) for a in model_variables)
        s = s.remap_inputs(ind)
        ge.connect(pv, s)

    logger.info('Graph connected')

    with tf.name_scope('optimize'):
        optimizer = tf.train.AdamOptimizer(FLAGS.meta_lr)
        update_step = optimizer.minimize(meta_loss)

    def gen_data(phase):
        x_batch_update = np.random.uniform(-1, 1, size=(FLAGS.meta_batch_size, n_batch,))
        x_batch_meta = np.random.uniform(-1, 1, size=(FLAGS.meta_batch_size, n_batch,))
        y_batch_update = np.sin(x_batch_update + phase)
        y_batch_meta = np.sin(x_batch_meta + phase)
        return x_batch_update, y_batch_update, x_batch_meta, y_batch_meta

    board_writer = tf.summary.FileWriter('tensorboard_log', tf.get_default_graph())
    n_batch = 10
    print_every = 10

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for j in range(1000):
            phase = np.random.uniform(0, np.pi, size=FLAGS.meta_batch_size)[:, None]
            x_batch_update, y_batch_update, x_batch_meta, y_batch_meta = gen_data(phase)

            feed_dict = {
                x_update_: x_batch_update[:, :, None],
                x_meta_: x_batch_meta[:, :, None],
                y_update_: y_batch_update[:, :, None],
                y_meta_: y_batch_meta[:, :, None]
            }

            session.run(update_step, feed_dict=feed_dict)

            if j % print_every == 0:
                phase = np.random.uniform(0, np.pi, size=FLAGS.meta_batch_size)[:, None]
                x_batch_update, y_batch_update, x_batch_meta, y_batch_meta = gen_data(phase)

                feed_dict = {
                    x_update_: x_batch_update[:, :, None],
                    x_meta_: x_batch_meta[:, :, None],
                    y_update_: y_batch_update[:, :, None],
                    y_meta_: y_batch_meta[:, :, None]
                }
                first_update_losses = list(a[0] for a in update_losses)
                u, m = session.run([first_update_losses, meta_loss], feed_dict=feed_dict)
                print('{:.4g} | {:.4g}'.format(np.mean(u), np.mean(m)))


if __name__ == '__main__':
    main()
