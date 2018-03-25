import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


class MAMLWrapper(object):
    def __init__(self, n_update_steps, update_learning_rate, stop_gradients=False, name='maml'):
        self.n_update_steps = n_update_steps
        self.update_lr = update_learning_rate
        self.name = name
        self.stop_gradients = stop_gradients
        self.x_update_ = None
        self.x_meta_ = None
        self.y_update_ = None

    def __call__(self, model, update_loss_function, x_update_=None, y_update_=None, x_meta_=None):
        updated_parameters = list()
        update_losses = list()

        non_model_trainable_variables = tf.trainable_variables()
        model_variables = []

        if x_meta_ is None:
            x_meta_ = tf.placeholder(dtype=tf.float32, shape=(None, x_update_.get_shape[1]))

        self.x_update_ = x_update_
        self.y_update_ = y_update_
        self.x_meta_ = x_meta_

        with tf.variable_scope(self.name):
            for i in range(self.n_update_steps):
                with tf.name_scope('update_{}'.format(i)):
                    with tf.variable_scope('model', reuse=(i != 0)):
                        y_update = model(x_update_)
                    if i == 0:
                        # __________________________________________________________________________
                        # determine the models variables
                        model_variables = list(item for item in tf.trainable_variables()
                                               if item not in non_model_trainable_variables)
                        updated_parameters.append(model_variables)
                    update_loss = update_loss_function(y_update, y_update_)
                    update_losses.append(update_loss)
                    # ______________________________________________________________________________
                    # ATTENTION:
                    # need to differentiate wrt the original parameters, since the updated model
                    # steps are still connected to the original parameters.
                    grads = tf.gradients(update_loss, model_variables)
                    if self.stop_gradients:
                        grads = list(tf.stop_gradient(grad) for grad in grads)
                    updated_parameters.append(
                        list(v - self.update_lr * g for v, g in zip(updated_parameters[-1], grads)))

            with tf.name_scope('meta'):
                with tf.variable_scope('model', reuse=True):
                    y_meta = model(x_meta_)

            # ______________________________________________________________________________________
            # connect updated parameters to each update step
            for i in range(1, self.n_update_steps):
                pv = ge.sgv(*updated_parameters[i])
                s = ge.sgv_scope('{}/update_{}'.format(self.name, i), tf.get_default_graph())
                ind = list(s.input_index(tf.convert_to_tensor(a)) for a in model_variables)
                s = s.remap_inputs(ind)
                ge.connect(pv, s)

            # _____________________________________________________________________________________
            # connect final updated parameters
            pv = ge.sgv(*updated_parameters[-1])
            s = ge.sgv_scope('{}/meta'.format(self.name), tf.get_default_graph())
            ind = list(s.input_index(tf.convert_to_tensor(a)) for a in model_variables)
            s = s.remap_inputs(ind)
            ge.connect(pv, s)

        # _________________________________________________________________________________________
        # for debug and evaluation purposes
        self.update_losses = update_losses
        self.updated_parameters = updated_parameters
        return y_meta

    def meta_optimize(self, optimizer, meta_loss, meta_batch_size=1):
        """
        Build the optimization procedure that enables 'pseudo' batch training.
        This is a hacky workaround.

        :param optimizer: The tensorflow optimizer used for meta learning. Needs to support compute_gradients and
        apply_gradients
        :param meta_loss: The loss objective to be minimized
        :param meta_batch_size: Meta batch size that scales the accumulated gradient
        :return: gradient accumulation op, update op
        """
        gvs = optimizer.compute_gradients(meta_loss)
        gradient_update = dict()
        gradient_zero_op = dict()
        gradient_accumulators = dict()
        batch_gvs = []
        # __________________________________________________________________________________________
        # hacky workaround to have batches of different tasks
        # necessary as the graph updates ONE set of parameters which would all be modified in
        # different directions in the setting of multiple tasks
        for g, v in gvs:
            # ______________________________________________________________________________________
            # skip variables that have no gradient to the meta loss
            if g is None:
                continue
            z = tf.zeros(g.get_shape())
            gradient_accumulator = tf.Variable(z, trainable=False)
            gradient_accumulators[v] = gradient_accumulator
            gradient_zero_op[v] = tf.assign(gradient_accumulator, z)
            gradient_update[v] = tf.assign_add(gradient_accumulator, g / meta_batch_size)
            batch_gvs.append((gradient_accumulator, v))
        update_step = optimizer.apply_gradients(batch_gvs)

        gradient_accumulate = list(gradient_update.values())
        optimize_step = [update_step]
        optimize_step.extend(list(gradient_zero_op.values()))
        return gradient_accumulate, optimize_step
