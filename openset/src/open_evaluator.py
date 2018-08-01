from __future__ import print_function

from model.ops import *
import random
import cv2
from PIL import Image
import os, sys
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from glob import glob
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from dataio.data_reader import TestReader, CLSReader
from model.base_model import SimpleModel
from sklearn.metrics import confusion_matrix as sk_conf_mat

F = tf.app.flags.FLAGS
r2d = 180.0/3.14

class GazeEval():
    def __init__(self):
        self.dataloader = TestReader()
        self.build_model()

        self.test_data = self.dataloader.create_test_dataset()
        self.test_iter = self.test_data.make_one_shot_iterator()

    def get_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels))

        self.out_class = tf.argmax(self.out, axis=1)
        self.labels_class = tf.argmax(self.labels, axis=1)

        def compute_weights(x):
            return tf.cond(tf.equal(x, 12), lambda: tf.constant(1/10.), lambda: tf.constant(1.0))

        class_weights = tf.map_fn(compute_weights, self.labels_class, dtype = tf.float32)
        self.weighted_loss = tf.losses.softmax_cross_entropy(onehot_labels = self.labels,
                                                            logits = self.out,
                                                            weights = class_weights)

        self.accuracy = tf.contrib.metrics.accuracy(predictions = self.out_class,
                                                     labels = self.labels_class)

        self.mean_class_wise_accuracy, self.mean_class_wise_accuracy_update = tf.metrics.mean_per_class_accuracy(predictions = self.out_class, 
                                                                    labels = self.labels_class, 
                                                                    num_classes = F.output_dim)
        
        self.confusion_matrix = tf.confusion_matrix(labels = self.labels_class,
                                                    predictions = self.out_class, 
                                                    num_classes = F.output_dim)

    def build_model(self):
        self.images, self.labels, self.filenames = self.dataloader.get_model_inputs()

        self.labels = tf.cast(self.labels, tf.int32)    #Small Hack for converting 

        model = SimpleModel(self.images, self.labels, output_dim=F.output_dim, scope='source_classifier')
        self.out, _ = model.get_model()
        self.get_loss()

    def print_evaluation_metrics(self, eval_confusion_matrix, eval_loss, eval_wloss, eval_accuracy, eval_class_accuracy):
        eval_loss = np.array(eval_loss)
        eval_wloss = np.array(eval_wloss)
        eval_accuracy = np.array(eval_accuracy)
        eval_class_accuracy = np.array(eval_class_accuracy)
        logging.info("Evaluation Metrics  #################")
        logging.info("Current Evaluation Loss: {}, Mean Loss: {}, Mean Weighted-Loss: {}, \
            Mean Accuracy: {},  Mean Class-Wise Accuracy: {}".format(len(eval_loss), 
            eval_loss.mean(), eval_wloss.mean(), eval_accuracy.mean(), eval_class_accuracy.mean()))

        total_labels = np.sum(eval_confusion_matrix, axis=1)
        correct_preds = np.diag(eval_confusion_matrix)
        eval_class_wise_accuracy = 1.*correct_preds/total_labels
        known_mean_accuracy = np.mean(eval_class_wise_accuracy[:-1])
        logging.info("Mean Class-wise Accuracy: {}, Mean Known Accuracy: {}".format(eval_class_wise_accuracy, known_mean_accuracy))

    def eval(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Testing Source Network")

        tf.summary.scalar('cross_entropy_loss', self.loss)
        tf.summary.scalar('Weighted_Loss', self.weighted_loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('mean_class_wise_accuracy', self.mean_class_wise_accuracy)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            print("Log-dir: ", F.checkpoint_dir)
            return self.saver.restore(sess, F.checkpoint_dir + F.checkpoint_file)

        self.test_handle_op = self.test_iter.string_handle()

        # Define your supervisor for running a managed session.
        sv = tf.train.Supervisor(logdir=F.log_eval_dir, init_fn=restore_fn, summary_op=None, saver=None)

        submission_file = open(F.submission_file, 'w')

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    
            logging.info('Starting evaluation  in open_evaluator.py: ')
            self.test_handle = sess.run(self.test_handle_op)
            # sess.run(self.test_iter.initializer)
            eval_loss, eval_wloss, eval_accuracy, eval_class_accuracy = [], [], [], []
            eval_confusion_matrix = None
            while True:
                try:
                    out, loss, wloss, accuracy, _, class_wise_accuracy, confusion_matrix, labels, filenames = sess.run([self.out_class, self.loss, self.weighted_loss, self.accuracy, 
                                self.mean_class_wise_accuracy_update, self.mean_class_wise_accuracy,
                                self.confusion_matrix, self.labels_class, self.filenames], 
                                feed_dict={self.dataloader.split_handle: self.test_handle})
                    # logging.info("Trying...{}, mean label: {}".format(len(eval_loss), np.mean(labels)))
                    eval_loss.append(loss)
                    eval_wloss.append(wloss)
                    eval_accuracy.append(accuracy)
                    eval_class_accuracy.append(class_wise_accuracy)

                    if eval_confusion_matrix is not None:
                        eval_confusion_matrix += np.array(confusion_matrix)
                    else:
                        eval_confusion_matrix = np.array(confusion_matrix)

                    submission_file.write("\n".join(map(str, out.tolist())))
                    submission_file.write("\n")
                except:
                    submission_file.close()
                    print("Metrics calculated")
                    if len(eval_loss) != 0:
                        eval_loss = np.array(eval_loss)
                        eval_wloss = np.array(eval_wloss)
                        eval_accuracy = np.array(eval_accuracy)
                        eval_class_accuracy = np.array(eval_class_accuracy)
                        
                        self.print_evaluation_metrics(eval_confusion_matrix, eval_loss, eval_wloss, eval_accuracy, eval_class_accuracy)
                    break