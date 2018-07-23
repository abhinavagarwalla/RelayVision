from __future__ import print_function

from model.ops import *
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from dataio.data_reader import CLSReader
from model.base_model import SimpleModel

F = tf.app.flags.FLAGS

class OpensetClassifier():
    def __init__(self):
        self.dataloader = CLSReader()

        self.build_model()

        self.training_data = self.dataloader.create_training_dataset()
        self.validation_data = self.dataloader.create_validation_dataset()

        self.training_iter = self.training_data.make_one_shot_iterator() #make_initializable_iterator()
        self.validation_iter = self.validation_data.make_initializable_iterator()

    def get_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels))

        out_class = tf.argmax(self.out, axis=1)
        labels_class = tf.argmax(self.labels, axis=1)

        def compute_weights(x):
            return tf.cond(tf.equal(x, 12), lambda: tf.constant(1/10.), lambda: tf.constant(1.0))

        class_weights = tf.map_fn(compute_weights, labels_class, dtype = tf.float32)
        self.weighted_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels,
                                                            logits = self.out,
                                                            weights = class_weights)

        self.accuracy = tf.contrib.metrics.accuracy(predictions = out_class,
                                                     labels = labels_class)

        self.mean_class_wise_accuracy, self.mean_class_wise_accuracy_update = tf.metrics.mean_per_class_accuracy(predictions = out_class, 
                                                                    labels = labels_class, 
                                                                    num_classes = F.output_dim)
        
        self.confusion_matrix = tf.confusion_matrix(labels = labels_class,
                                                    predictions = out_class, 
                                                    num_classes = F.output_dim)

    def build_model(self):
        self.images, self.labels = self.dataloader.get_model_inputs()

        self.labels = tf.cast(self.labels, tf.int32)    #Small Hack for converting 

        model = SimpleModel(self.images, self.labels, output_dim=F.output_dim, scope='source_classifier')
        self.out, _ = model.get_model()
        print(self.images.get_shape(), self.labels.get_shape(), self.out.get_shape())
        self.get_loss()        

    def train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Training Source Network")

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        self.lr = tf.train.exponential_decay(
            learning_rate=F.initial_learning_rate,
            global_step=global_step,
            decay_steps=F.decay_step,
            decay_rate=F.decay_rate,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        self.grad_update = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.weighted_loss,
                                                                                 global_step=global_step)

        tf.summary.scalar('cross_entropy_loss', self.loss)
        tf.summary.scalar('learning_rate', self.lr)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('class_wise_accuracy', self.mean_class_wise_accuracy)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            return self.saver.restore(sess, F.checkpoint_dir + F.checkpoint_file)

        self.training_handle_op = self.training_iter.string_handle()
        self.validation_handle_op = self.validation_iter.string_handle()

        # Define your supervisor for running a managed session.
        if F.load_chkpt:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, init_fn=restore_fn, saver=self.saver)
        else:
            sv = tf.train.Supervisor(logdir=F.log_dir, summary_op=None, init_fn=None, saver=self.saver)

        current_best_loss = 1000. #TODO: Read it from a file for multiple restarts
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            self.training_handle = sess.run(self.training_handle_op)
            self.validation_handle = sess.run(self.validation_handle_op)

            for step in range(int(F.num_steps)):
                #try:
                if step % F.log_every == 0:
                    loss, wloss, _, accuracy, _1, class_wise_accuracy, confusion_matrix, summaries, global_step_count = \
                            sess.run([self.loss, self.weighted_loss, self.grad_update, self.accuracy, 
                            self.mean_class_wise_accuracy_update, self.mean_class_wise_accuracy,
                            self.confusion_matrix, self.summary_op, sv.global_step], 
                            feed_dict={self.dataloader.split_handle: self.training_handle})

                    sv.summary_computed(sess, summaries, global_step=global_step_count)
                    logging.info("Step: {}/{}, Global Step: {}, loss: {}, wloss: {}, accuracy: {}, \
                        class-wise accuracy: {}".format(step, F.num_steps, global_step_count, 
                        loss, wloss, accuracy, class_wise_accuracy))
                    logging.info(confusion_matrix)
                else:
                    loss, wloss, _,  global_step_count = sess.run([self.loss, self.weighted_loss,
                            self.grad_update, sv.global_step], feed_dict={self.dataloader.split_handle: self.training_handle})
                #except:
                #    logging.info("Smaller batch size error,.. proceeding to next batch size")
                    # pass

                # # logging.info("A step taken")
                if step % F.save_every==1:
                    logging.info('Saving model to disk as step={}'.format(step))
                    sess.run(self.validation_iter.initializer)
                    eval_loss, eval_wloss, eval_accuracy, eval_class_accuracy = [], [], [], []
                    while True:
                        try:
                            loss, wloss, accuracy, class_wise_accuracy, labels = sess.run([self.loss, self.weighted_loss, self.accuracy, 
                                self.mean_class_wise_accuracy,  self.labels], 
                                feed_dict={self.dataloader.split_handle: self.validation_handle})
                            # logging.info("Trying...{}, mean label: {}".format(len(eval_loss), np.mean(labels)))
                            eval_loss.append(loss)
                            eval_wloss.append(wloss)
                            eval_accuracy.append(accuracy)
                            eval_class_accuracy.append(class_wise_accuracy)
                        except:
                            if len(eval_loss) != 0:
                                eval_loss = np.array(eval_loss)
                                eval_wloss = np.array(eval_wloss)
                                eval_accuracy = np.array(eval_accuracy)
                                eval_class_accuracy = np.array(eval_class_accuracy)
                                logging.info("Current Evaluation Loss at step({}): {}, Mean Loss: {}, Mean Weighted-Loss: {}, \
                                    Mean Accuracy: {},  Mean Class-Wise Accuracy: {}".format(step, len(eval_loss), 
                                    eval_loss.mean(), eval_wloss.mean(), eval_accuracy.mean(), eval_class_accuracy.mean()))
                            if eval_wloss.mean() < current_best_loss:
                                print('tada.. lower weighted eval loss!!!')
                                current_best_loss = eval_wloss.mean()
                                sv.saver.save(sess, sv.save_path+'_reducedLoss', global_step=global_step_count)
                            break