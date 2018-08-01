import os
import sys
import scipy.misc
import numpy as np
import pprint
sys.path.insert(0,'../')
from src.open_model_classifier import OpensetClassifier
from src.open_evaluator import GazeEval
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("num_steps", 50000, "Epochs to train ")
flags.DEFINE_integer("decay_step", 5000, "Decay step of learning rate in steps")
flags.DEFINE_float("decay_rate", 0.5, "Decay rate of learning rate")
flags.DEFINE_float("gpu_frac", 0.5, "Gpu fraction")
flags.DEFINE_float("initial_learning_rate", 0.001, "Learing rate")

# Data paths for Aggie's laptop
flags.DEFINE_string("train_data_path", "../data/train_tfrecords/", "Directory name containing the dataset [data]")
flags.DEFINE_string("val_data_path", "../data/validation_tfrecords/", "Directory name containing the dataset [data]")
flags.DEFINE_string("test_data_path", "../data/validation_tfrecords/", "Directory name containing the dataset [data]")

# Data paths for Arna's lab PC
# flags.DEFINE_string("train_data_path", "/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/RelayVision/train_tfrecords/", "Directory name containing the dataset [data]")
# flags.DEFINE_string("val_data_path", "/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/RelayVision/validation_tfrecords/", "Directory name containing the dataset [data]")
# flags.DEFINE_string("test_data_path", "/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/RelayVision/test_tfrecords/", "Directory name containing the dataset [data]")

flags.DEFINE_boolean("evaluate", False, "Whether to evaluate a checkpoint or train?")
flags.DEFINE_string("log_eval_dir", "../logs/eval/", "Directory name to save the logs [logs]")

flags.DEFINE_string("submission_file", "../results/submission_1.txt", "The submission file to save predictions")

flags.DEFINE_boolean("visualise", False, "Whether to visualise predictions?")
flags.DEFINE_string("visualise_dir", "../vis/", "Directory to store visualisations")

flags.DEFINE_string("checkpoint_dir", "../checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_file", "model.ckpt_reducedLoss", "Name of the model checkpoint")
flags.DEFINE_string("log_dir", "../logs/open-try-1/", "Directory name to save the logs [logs]")
flags.DEFINE_boolean("load_chkpt", False, "True for loading saved checkpoint")

flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("output_dim", 13, "Number of output classes.")

flags.DEFINE_integer("channels", 3, "Number of channels in input image")
flags.DEFINE_integer("img_height", 64, "Height of Input Image")
flags.DEFINE_integer("img_width", 64, "Height of Input Image")

flags.DEFINE_integer("capacity", 8*10, "Capacity of input queue")
flags.DEFINE_integer("num_threads", 4, "Threads to employ for filling input queue")
flags.DEFINE_integer("min_after_dequeue", 8, "Minimum samples to remain after dequeue")

flags.DEFINE_integer("log_every", 50, "Frequency of logging for summary")
flags.DEFINE_integer("save_every", 1000, "Save after steps")
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    if not os.path.exists(FLAGS.log_eval_dir):
        os.makedirs(FLAGS.log_eval_dir)


    if not FLAGS.evaluate:
        source_task = OpensetClassifier()
        source_task.train()
    else:
        eval_task = GazeEval()
        eval_task.eval()

if __name__ == '__main__':
    tf.app.run()
