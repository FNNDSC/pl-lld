
import os
import tensorflow as tf
import numpy as np
import sys
from LLDcode.tensorflow_train.utils.summary_handler import SummaryHandler, create_summary_placeholder
from LLDcode.utils.io.common import create_directories, copy_files_to_folder
import datetime
from collections import OrderedDict
from glob import glob

class MainLoopBase(object):
    def __init__(self):
        self.saver = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self.first_iteration = True
        self.train_queue = None
        self.val_queue = None
        self.batch_size = None
        self.learning_rate = None
        self.optimizer = None
        self.optimization_function = None
        self.current_iter = 0
        self.disp_iter = 1
        self.layer_weight_summary_iter = None
        self.layer_weight_inspector = None
        self.max_iter = None
        self.snapshot_iter = None
        self.test_iter = None
        self.test_initialization = True
        self.train_losses = None
        self.val_losses = None
        self.is_closed = False
        self.output_folder = ''
        self.load_model_filename = None
        self.files_to_copy = ['*.py']
        self.additional_summaries_placeholders_val = None
        self.raise_on_nan_loss = True
        self.loss_name_for_nan_loss_check = 'loss'

    def init_saver(self):
        # initialize variables
        self.saver = tf.train.Saver(max_to_keep=1000)

    def init_variables(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print('Variables')
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)  # i.name if you want just a name

    def start_threads(self):
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        if self.train_queue is not None:
            self.train_queue.start_threads(self.sess)
        if self.val_queue is not None:
            self.val_queue.start_threads(self.sess)

    def load_model(self):
        if self.load_model_filename is not None:
            model_filename = self.load_model_filename
        else:
            #model_filename = os.path.join(self.output_folder, 'weights/model-15000')
            model_filename = os.path.join(self.output_folder, 'weights/model-' + str(self.current_iter))
        print('Restoring model ' + model_filename)
        self.restore_variables(self.sess, model_filename)

    def restore_variables(self, session, model_filename):
        self.saver.restore(session, model_filename)

    def optimistic_restore(self, session, model_filename, except_var_names=None):
        if except_var_names is None:
            except_var_names = []
        reader = tf.train.NewCheckpointReader(model_filename)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()])
        name_var_dict = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        restore_vars = []
        not_restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name_var_dict[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if saved_var_name in saved_shapes and var_shape == saved_shapes[saved_var_name] and var_name not in except_var_names:
                    restore_vars.append(curr_var)
                else:
                    not_restore_vars.append(var_name)
        print('not restoring', not_restore_vars)
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, model_filename)

    def create_output_folder(self):
        create_directories(self.output_folder)
        if self.files_to_copy is not None:
            all_files_to_copy = []
            for file_to_copy in self.files_to_copy:
                all_files_to_copy += glob(file_to_copy)
            copy_files_to_folder(all_files_to_copy, self.output_folder)

    def output_folder_timestamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def output_folder_for_current_iteration(self):
        return os.path.join(self.output_folder, 'iter_' + str(self.current_iter))

    def output_file_for_current_iteration(self, *args):
        return os.path.join(self.output_folder, 'iter_' + str(self.current_iter), *args)

    def init_all(self):
        self.init_networks()
        self.initLossAggregators()
        self.init_variables()
        self.start_threads()
        self.init_saver()
        self.create_output_folder()

    def stop_threads(self):
        self.coord.request_stop()
        if self.train_queue is not None:
            self.train_queue.close(self.sess)
        if self.val_queue is not None:
            self.val_queue.close(self.sess)
        self.coord.join(self.threads)

    def snapshot(self):
        print('Creating snapshot...')
        save_path = self.saver.save(self.sess, os.path.join(self.output_folder, 'weights/model'), global_step=self.current_iter)
        print('Model saved in file %s' % save_path)

    def train(self):
        """
        Run optimizer, loss, layer_weight_summary and update loss accumulators
        """
        if self.optimization_function is not None:
            fetches = (self.optimization_function,)
        else:
            fetches = (self.optimizer,)
        # add layer_weight_inspector summaries, if set
        if self.layer_weight_inspector is not None and (self.current_iter % self.layer_weight_summary_iter) == 0:
            fetches += (self.layer_weight_inspector.summary,)
        # add train_loss_aggregator update ops, if set
        if self.train_loss_aggregator is not None:
            fetches = fetches + self.train_loss_aggregator.get_update_ops()
        # add the update function of the train_queue, if set
        if self.train_queue.update() is not None:
            fetches = fetches + (self.train_queue.update(),)
            results = self.sess.run(fetches)
        print("fetches",fetches)
        # run fetches
        results = self.sess.run(fetches)

        # display loss and save summary
        if self.train_loss_aggregator is not None and (self.current_iter % self.disp_iter) == 0:
            summary_values = self.train_loss_aggregator.finalize(self.current_iter)
            # check if current loss is nan and if training should be stopped
            if self.raise_on_nan_loss and self.loss_name_for_nan_loss_check in summary_values:
                if np.isnan(summary_values[self.loss_name_for_nan_loss_check]):
                    raise RuntimeError('\'{}\' is nan'.format(self.loss_name_for_nan_loss_check))
        # save layer_weight_summary
        if self.layer_weight_inspector is not None and (self.current_iter % self.layer_weight_summary_iter) == 0:
            summary = results[1]  # TODO: make more flexible, currently summary is always entry 1
            self.layer_weight_inspector.writer.add_summary(summary, global_step=self.current_iter)

    def print_training_parameters(self):
        print('Training parameters:')
        if isinstance(self.optimizer, tf.train.GradientDescentOptimizer):
            print('Optimizer: SGD')
        elif isinstance(self.optimizer, tf.train.MomentumOptimizer):
            print('Optimizer: momentum')
        elif isinstance(self.optimizer, tf.train.AdamOptimizer):
            print('Optimizer: adam')
        if self.batch_size is not None:
            print('Batch size:', self.batch_size)
        if self.learning_rate is not None:
            print('Learning rate:', self.learning_rate)
        if self.max_iter is not None:
            print('Max iterations:', self.max_iter)
        if self.output_folder is not None:
            print('Output folder:', self.output_folder)

    def run(self):
        print("*************INIT ALL****************")
        self.init_all()
        if self.current_iter > 0 or self.load_model_filename is not None:
            print("*************LOADING MODEL****************")
            self.load_model()
        print('Starting main loop')
        self.print_training_parameters()
        try:
            while self.current_iter <= self.max_iter:
                # snapshot
                if (self.current_iter % self.snapshot_iter) == 0 and not self.first_iteration:
                    self.snapshot()
                # test
                if (self.current_iter % self.test_iter) == 0 and (self.test_initialization or not self.first_iteration):
                    self.test()
                # do not train in last iteration
                if self.current_iter < self.max_iter:
                    print("Run training")
                    self.train()
                self.current_iter += 1
                self.first_iteration = False
                sys.stdout.flush()
        finally:
            self.close()

    def run_test(self):
        self.init_all()
        self.load_model()
        print('Starting main test loop')
        try:
            self.test()
        finally:
            self.close()

    def initLossAggregators(self):
        if self.train_losses is not None and self.val_losses is not None:
            assert set(self.train_losses.keys()) == set(self.val_losses.keys()), 'train and val loss keys are not equal, ' + ', '.join(map(str, self.train_losses.keys())) + ' and ' + ', '.join(map(str, self.val_losses.keys()))

        if self.train_losses is None:
            return

        summaries_placeholders = OrderedDict([(loss_name, create_summary_placeholder(loss_name)) for loss_name in self.train_losses.keys()])

        # mean values used for summaries
        self.train_loss_aggregator = SummaryHandler(self.sess,
                                                    self.train_losses,
                                                    summaries_placeholders,
                                                    'train',
                                                    os.path.join(self.output_folder, 'train'),
                                                    os.path.join(self.output_folder, 'train.csv', ))

        if self.val_losses is None:
            return

        summaries_placeholders_val = summaries_placeholders.copy()

        if self.additional_summaries_placeholders_val is not None:
            summaries_placeholders_val.update(self.additional_summaries_placeholders_val)

        self.val_loss_aggregator = SummaryHandler(self.sess,
                                                  self.val_losses,
                                                  summaries_placeholders_val,
                                                  'test',
                                                  os.path.join(self.output_folder, 'test'),
                                                  os.path.join(self.output_folder, 'test.csv'))

    def init_networks(self):
        print('initNetworks() is deprecated and may be removed in later versions. Use init_networks() instead.')
        self.initNetworks()

    def initNetworks(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def close(self):
        if not self.is_closed:
            self.stop_threads()
            self.sess.close()
            tf.reset_default_graph()
            self.is_closed = True
