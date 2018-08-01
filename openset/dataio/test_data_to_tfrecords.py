from __future__ import print_function
import glob
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tqdm import tqdm
import numpy as np
from random import shuffle, random
from os.path import expanduser
from pathos.multiprocessing import ProcessPool as Pool
from scipy.io import loadmat
from scipy.misc import imresize, imread
import cv2

import tensorflow as tf

def _float_feature(value):
  return tf.train.Feature(int64_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

label_mapping = {'aeroplane': 0, 'bicycle': 1, 'bus': 2, 'car': 3,
    'horse': 4, 'knife': 5, 'motorcycle': 6, 'person': 7, 'plant': 8,
    'skateboard': 9, 'train': 10, 'truck': 11, 'other': 12}
classes = 13

def load_image_label(img_data_path, addr):
    img = cv2.resize(cv2.imread(img_data_path + addr), (64, 64))
    # print(img.shape)
    label = np.zeros(classes)
    label[label_mapping[addr.split('/')[-2]]] = 1
    return img.astype(np.float32), np.array(label).astype(np.float32)

def write_record(img_data_path, train_filename, addrs, split):
    # print(train_filename, img_data_path)
    writer = tf.python_io.TFRecordWriter(train_filename)
    for idx in tqdm(range(len(addrs))):
        # try:
        img, label = load_image_label(img_data_path, addrs[idx])

        # print(img.shape, label.shape)
        # Create a feature
        feature = {split + '/label': _bytes_feature(tf.compat.as_bytes(label.tostring())),
                   split + '/image': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                   split + '/filename': _bytes_feature(tf.compat.as_bytes(addrs[idx].tostring()))}
        
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # except:
        #     print("Couldn't write to tf, exception raised..moving to next image")

    writer.close()

split_str = 'validation'
def write_data(img_data_path):
    categories = filter(lambda x: os.path.isdir(os.path.join(img_data_path,x)), os.listdir(img_data_path))

    addrs = [i.strip().split(' ')[0] for i in open(img_data_path + 'image_list.txt').readlines()]
    print(len(addrs))
    n_shards = 16
    addrs = np.array_split(np.array(addrs), n_shards)

    write_data_path = expanduser("~") + '/Desktop/RelayVision/data/' + split_str + '_tfrecords/' # for Aggie's laptop
    # write_data_path = '/home/arna/Project/RelayVision/' + split_str + '_tfrecords/' # for Arna's lab PC
    filenames = [write_data_path+'{}_{:0>3}_{:0>3}.tfrecords'.format(split_str, i, n_shards-1) for i in range(n_shards)]

    # for i in range(len(filenames)):
    i=0
    write_record(img_data_path, filenames[i], addrs[i], split_str)
    # p = Pool(n_shards)
    # p.map(write_record, [img_data_path for i in range(n_shards)], filenames, addrs, [split_str for i in range(n_shards)])
    # sys.stdout.flush()

if __name__=="__main__":
    home = expanduser("~")
    img_data_path = home + '/Desktop/RelayVision/data/' + split_str + os.sep # for Aggie's laptop    
    # img_data_path = '/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/RelayVision/validation/' #+ split_str + os.sep #for Arna's lab PC
    write_data(img_data_path)
