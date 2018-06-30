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
F = tf.app.flags.FLAGS
r2d = 180.0/3.14

class GazeEval():
    def __init__(self):
        self.initialise_tf_reader(split='test')
        self.build_model()
 
    def angle_disparity(self, v1, v2):
        radial_distance = np.sum(v1 * v2)
        return np.arccos(radial_distance) * r2d

    def initialise_tf_reader(self, split='train'):
        feature = {split + '/image': tf.FixedLenFeature([], tf.string),
                split + '/label': tf.FixedLenFeature([], tf.string)}

         # Create a list of filenames and pass it to a queue
        filenames = glob(F.val_data_path + split + '*.tfrecords')
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=1)
        
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        # Creates batches by randomly shuffling tensors
        self.images, self.labels = tf.train.shuffle_batch([image, label], batch_size=F.batch_size, capacity=F.capacity,
         num_threads=F.num_threads, min_after_dequeue=F.min_after_dequeue)

    def build_model(self):
        with tf.variable_scope('source_classifier'):
            input0 = normalize(self.images)
            h0 = lrelu(conv2d(input0, 32, 3, 3, 1, 1, name='h0_conv'), 0.2)
            h1 = lrelu(conv2d(h0, 32, 3, 3, 1, 1, name='h1_conv'), 0.2)
            h2 = lrelu(conv2d(h1, 64, 3, 3, 1, 1, name='h2_conv'), 0.2)
            
            p1 = tf.layers.max_pooling2d(h2, 3, 2, name='pool1')
            h3 = lrelu(conv2d(p1, 80, 3, 3, 1, 1, name='h3_conv'), 0.2)
            h4 = lrelu(conv2d(h3, 192, 3, 3, 1, 1, name='h4_conv'), 0.2)

            p2 = tf.layers.max_pooling2d(h4, 2, 2, name='pool2')
            l1 = tf.contrib.layers.flatten(p2)
            l2 = lrelu(linear(l1, 9600, scope="Linear9600"), 0.2)
            l3 = lrelu(linear(l2, 1000, scope="Linear1000"), 0.2)
            #self.out = tf.nn.tanh(linear(l3, F.output_dim))
            self.out = linear(l3, F.output_dim)
        
        # xl, yl, zl = tf.split(self.labels, 3, axis=1)
        # xo, yo, zo = tf.split(self.out, 3, axis=1)
        # thetal, thetao = tf.asin(-yl), tf.asin(-yo)
        # phil, phio = tf.atan2(-zl, -xl), tf.atan2(-zo, -xo)
        # self.lb = tf.concat([thetal, phil], axis=1)
        # self.ob = tf.concat([thetao, phio], axis=1)
        # self.loss2d = tf.scalar_mul(tf.constant(r2d), tf.losses.mean_squared_error(self.lb, self.ob, 2))
        
        self.lbl = tf.nn.l2_normalize(self.labels, 1)
        self.out = tf.nn.l2_normalize(self.out, 1)
        # print("** Label:", self.lbl.get_shape())
        # print("**out:", self.out.get_shape())
        self.losscos = r2d*tf.acos(1-tf.losses.cosine_distance(self.lbl, self.out, dim=1))
        self.loss = tf.losses.mean_squared_error(self.lbl, self.out)
        
        xll, yll, zll = tf.split(self.lbl, 3, axis=1)
        xoo, yoo, zoo = tf.split(self.out, 3, axis=1)
        self.loss_x = tf.losses.mean_squared_error(xll,  xoo)
        self.loss_y = tf.losses.mean_squared_error(yll,  yoo)
        self.loss_z = tf.losses.mean_squared_error(zll,  zoo)
    
    def get2dlossnp(self, labels, out):
        thetal, thetao = np.arcsin(-labels[:,1]), np.arcsin(-out[:,1])
        phil, phio = np.arctan2(-labels[:,2], -labels[:,0]), np.arctan2(-out[:,2], -out[:,0])
        loss2d = np.mean((thetal-thetao)**2 + (phil-phio)**2)
        return loss2d*r2d

    def eval(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("Training Source Network")

        tf.summary.scalar('mse_loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            print("Log-dir: ", F.checkpoint_dir)
            return self.saver.restore(sess, F.checkpoint_dir + F.checkpoint_file)

        # Define your supervisor for running a managed session.
        sv = tf.train.Supervisor(logdir=F.log_eval_dir, init_fn=restore_fn, summary_op=None, saver=self.saver)

        current_best_loss = 1000. #TODO: Read it from a file for multiple restarts
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=F.gpu_frac)
        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    
            logging.info('Starting evaluation: ')
            eval_loss = []
            while True:
                try:
                    if not F.visualise:
                        loss, lx, ly, lz, losscos = sess.run([self.loss, self.loss_x,  self.loss_y,  self.loss_z, self.losscos])
                        print("ckpt::", sv.save_path)
                        # loss, loss2d, losscos, labels, out = sess.run([self.loss, self.loss2d, self.losscos, self.labels, self.out])
                        # logging.info("Trying...{}, mean label difference: {}".format(len(eval_loss), np.mean(labels)-np.mean(out)))
                        # print("First sample: ")
                        #loss2dnp = self.get2dlossnp(labels, out)
                        logging.info("Batch loss3d: {}, losscos: {}, lx: {}, ly: {}, lz:{}".format(loss, losscos, lx, ly, lz))
                        eval_loss.append(losscos)
                    else:
                        loss, images, gts, preds, gts_labels, losscos = sess.run([self.loss, self.images, self.lbl, self.out, self.labels, self.losscos])
                        images = np.uint8(images)
                        for idx in range(len(images)):
                            #print("b4 norm: ", gts_labels[idx], "||", "after norm:", gts[idx])
                            eye_c = np.array([55/2, 35/2])
                            cv2.line(images[idx], tuple(eye_c), tuple(eye_c+(gts[idx,:2]*80).astype(int)), (0, 127, 127), 1)
                            cv2.line(images[idx], tuple(eye_c), tuple(eye_c+(preds[idx,:2]*80).astype(int)), (255,255,255), 1)                                
                            im = Image.fromarray(images[idx].reshape(35, 55))
                            a_loss = self.angle_disparity(gts[idx], preds[idx])
                            im.save(F.visualise_dir + os.sep + str(idx) + '__' +  str(a_loss) + '.jpg')
                            print("Gt:", gts[idx], "||", "  Pred: ", preds[idx], "||", "aloss: ", a_loss)
                        print("Vis done.")
                        sys.exit()
                except:
                    print("Exception Raised")
                    eval_loss = np.array(eval_loss)
                    if len(eval_loss):
                        print("Current Evaluation Loss: {}, {}, {}, {}".format(len(eval_loss), eval_loss.mean(), eval_loss.max(), eval_loss.min()))
                        if eval_loss.mean() < current_best_loss:
                            current_best_loss = eval_loss.mean()
                            sv.saver.save(sess, sv.save_path)
                    break
