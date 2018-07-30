import tensorflow as tf
from glob import glob

F = tf.app.flags.FLAGS

class CLSReader():
    def __init__(self):
        self.split_handle = tf.placeholder(tf.string, shape=[])
        self.train_split, self.val_split, self.test_split = None, None, None
        self.iterator = tf.data.Iterator.from_string_handle(self.split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))
        self.next_element = self.iterator.get_next()

        self.split = None
        self.test_parse = None

    def create_training_dataset(self):
        self.split = 'train'
        filenames = glob(F.train_data_path + 'train*.tfrecords')
        print(filenames)
        self.train_split = self.create_dataset(filenames)
        return self.train_split

    def create_validation_dataset(self):
        self.split = 'validation'
        filenames = glob(F.val_data_path + 'validation*.tfrecords')
        print(filenames)
        self.val_split = self.create_dataset(filenames, 1)
        return self.val_split

    def create_test_dataset(self):
        self.split = 'validation'
        filenames = glob(F.test_data_path + 'validation*.tfrecords')
        filenames.sort()
        print(filenames)
        self.test_split = self.create_dataset(filenames, 1)
        return self.test_split

    def create_dataset(self, filenames, num_epochs=None):
        # filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_records, num_parallel_calls=F.num_threads)#, output_buffer_size=F.capacity)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(F.batch_size)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        return dataset

    def parse_records(self, serialized_example):
        feature = {self.split + '/image': tf.FixedLenFeature([], tf.string),
                    self.split + '/label': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[self.split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[self.split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        return image, label

    def get_model_inputs(self):
        return self.next_element

class TestReader():
    def __init__(self):
        self.split_handle = tf.placeholder(tf.string, shape=[])
        self.test_split = None
        self.iterator = tf.data.Iterator.from_string_handle(self.split_handle,
         (tf.float32, tf.float32),
          ([F.batch_size, F.img_height, F.img_width, F.channels],
           [F.batch_size, F.output_dim]))
        self.next_element = self.iterator.get_next()

        self.split = None
        self.test_parse = None

    def create_test_dataset(self):
        ##currently taking validation as test set not available
        self.split = 'validation'
        filenames = glob(F.test_data_path + 'validation*.tfrecords')
        filenames.sort()
        print(filenames)
        self.test_split = self.create_dataset(filenames, 1)
        return self.test_split

    def create_dataset(self, filenames, num_epochs=None):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_records, num_parallel_calls=F.num_threads)#, output_buffer_size=F.capacity)
        dataset = dataset.batch(F.batch_size)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        return dataset

    def parse_records(self, serialized_example):
        feature = {self.split + '/image': tf.FixedLenFeature([], tf.string),
                    self.split + '/label': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[self.split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[self.split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        return image, label

    def get_model_inputs(self):
        return self.next_element

class ADDAReader():
    def __init__(self):
        self.source_split_handle = tf.placeholder(tf.string, shape=[])
        self.target_split_handle = tf.placeholder(tf.string, shape=[])
        self.source_split, self.target_split, self.test_split = None, None, None

        self.source_iterator = tf.contrib.data.Iterator.from_string_handle(self.source_split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))
        self.target_iterator = tf.contrib.data.Iterator.from_string_handle(self.target_split_handle,
         (tf.float32, tf.float32), ([F.batch_size, F.img_height, F.img_width, F.channels], [F.batch_size, F.output_dim]))

        self.next_source_element = self.source_iterator.get_next()
        self.next_target_element = self.target_iterator.get_next()

        self.split = None

    def create_source_dataset(self):
        self.split = 'train'
        filenames = glob(F.source_data_path + 'train*.tfrecords')
        print(filenames)
        self.source_split = self.create_dataset(filenames)
        return self.source_split

    def create_target_dataset(self):
        self.split = 'train'
        filenames = glob(F.target_data_path + 'train*.tfrecords')
        print(filenames)
        self.target_split = self.create_dataset(filenames)
        return self.target_split

    def create_val_dataset(self):
        self.split = 'val'
        filenames = glob(F.test_data_path + 'val*.tfrecords')[:2]
        print(filenames)
        self.test_split = self.create_dataset(filenames)
        return self.test_split

    def create_dataset(self, filenames, num_epochs=None):
        # filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parse_records, num_threads=F.num_threads, output_buffer_size=F.capacity)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(F.batch_size)
        if num_epochs:
            dataset = dataset.repeat(num_epochs)
        else:
            dataset = dataset.repeat()
        return dataset

    def parse_records(self, serialized_example):
        feature = {self.split + '/image': tf.FixedLenFeature([], tf.string),
                self.split + '/label': tf.FixedLenFeature([], tf.string)}

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features[self.split + '/image'], tf.float32)
        image = tf.reshape(image, [F.img_height, F.img_width, F.channels])

        label = tf.decode_raw(features[self.split + '/label'], tf.float32)
        label = tf.reshape(label, [F.output_dim])

        return image, label

    def get_source_model_inputs(self):
        return self.next_source_element

    def get_target_model_inputs(self):
        return self.next_target_element