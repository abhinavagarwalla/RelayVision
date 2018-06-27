from ops import *
import tensorflow as tf
import sys

class MeanReLUModel():
    def __init__(self, images, labels, output_dim=None, mean=None, prob=0.5, scope=None):
        self.images = images
        self.labels = labels
        self.output_dim = output_dim
        self.scope = scope
        self.mean = mean
        self.prob = prob

    def adaptation(self, h2, h4,l2):
        h2_shape = h2.get_shape().as_list()
        h4_reshaped = tf.image.resize_bilinear(h4, [32, 60])
        h2_reshaped = tf.image.resize_bilinear(h2, [32, 60])
        l2 = tf.reshape(l2, [-1, 32, 60, 5])
        a1 = tf.concat([h2_reshaped, h4_reshaped, l2], axis=3)
        #print("h2::", h2.get_shape())
        #print("h4:", h4.get_shape())
        #print("a1:", a1.get_shape())
        return a1

    def get_model(self):
        with tf.variable_scope(self.scope):
            input0 = mean_normalize(self.images, self.mean)
            # input0 = denormalize(input00)   #Converting from [-1, 1] to [0, 1]
            h0 = tf.nn.relu(conv2d(input0, 32, 3, 3, 1, 1, name='h0_conv'))
            h1 = tf.nn.relu(conv2d(h0, 32, 3, 3, 1, 1, name='h1_conv'))
            h2 = tf.nn.relu(conv2d(h1, 64, 3, 3, 1, 1, name='h2_conv'))
            
            p1 = tf.layers.max_pooling2d(h2, 3, 2, name='pool1')
            h3 = tf.nn.relu(conv2d(p1, 80, 3, 3, 1, 1, name='h3_conv'))
            h4 = tf.nn.relu(conv2d(h3, 192, 3, 3, 1, 1, name='h4_conv'))
            p2 = tf.layers.max_pooling2d(h4, 2, 2, name='pool2')
            l1 = tf.contrib.layers.flatten(p2)

            l2 = tf.nn.relu(linear(l1, 9600, scope="Linear9600"))
            l2 = tf.layers.dropout(inputs=l2, rate=self.prob, name='l2_dropout')
            l3 = tf.nn.relu(linear(l2, 1000, scope="Linear1000"))
            l3 = tf.layers.dropout(inputs=l3, rate=self.prob, name='l3_dropout')
            # out = tf.nn.tanh(linear(l3, F.output_dim))
            out = linear(l3, self.output_dim)

            to_adapt = self.adaptation(h2, h4, l2)
            return out, to_adapt

class SimpleModel():
    def __init__(self, images, labels, output_dim=None, scope=None):
        self.images = images
        self.labels = labels
        self.output_dim = output_dim
        self.scope = scope

    def adaptation(self, h2, h4,l2):
        h2_shape = h2.get_shape().as_list()
        h4_reshaped = tf.image.resize_bilinear(h4, [32, 60])
        h2_reshaped = tf.image.resize_bilinear(h2, [32, 60])
        l2 = tf.reshape(l2, [-1, 32, 60, 5])
        a1 = tf.concat([h2_reshaped, h4_reshaped, l2], axis=3)
        #print("h2::", h2.get_shape())
        #print("h4:", h4.get_shape())
        #print("a1:", a1.get_shape())
        return a1

    def get_model(self):
        with tf.variable_scope(self.scope):
            input0 = normalize(self.images)
            # input0 = denormalize(input00)   #Converting from [-1, 1] to [0, 1]
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
            # out = tf.nn.tanh(linear(l3, F.output_dim))
            out = linear(l3, self.output_dim)

            to_adapt = self.adaptation(h2, h4, l2)
            return out, to_adapt

class AdversaryModel():
    def __init__(self, input_vec=None, scope=None, prob=None, reuse=False):
        self.scope = scope
        self.input = input_vec
        if prob is not None:
            self.fc_prob = prob[0]
            self.conv_prob = prob[1]
        self.reuse = reuse

    def get_model(self):
        with tf.variable_scope(self.scope):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            # l1 = lrelu(linear(self.input, 1024, scope="Linear250_1"), 0.2)
            # l2 = lrelu(linear(l1, 1024, scope="Linear250_2"), 0.2)
            # out = linear(l2, 2, scope="FinalDecision")
            # return out
           
            # with 3D discriminators

            shape = self.input.get_shape().as_list()
            self.input = tf.reshape(self.input, [-1,shape[3], shape[1], shape[2], 1])
            h1 = lrelu(conv3d(input_=self.input, out_channels=16, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h1_conv3d'), 0.2)
            h1 = tf.layers.dropout(inputs=h1, rate=self.conv_prob, name='conv1_drop')

            # self.conv_prob = tf.cond(tf.less(self.conv_prob, tf.constant(0.01)), lambda: self.conv_prob, lambda: tf.scalar_mul(self.conv_prob, tf.constant(2.5)))
            h2 = lrelu((conv3d(input_=h1, out_channels=32, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h2_conv3d')), 0.2)
            h2 = tf.layers.dropout(inputs=h2, rate=self.conv_prob, name='conv2_drop')
            h3 = lrelu((conv3d(input_=h2, out_channels=64, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h3_conv3d')), 0.2)
            h3 = tf.layers.dropout(inputs=h3, rate=self.conv_prob, name='conv3_drop')
            h4 = lrelu((conv3d(input_=h3, out_channels=128, k_d=3, k_h=3, k_w=3, s_d=1, s_h=2, s_w=2, name='h4_conv3d')), 0.2)
            h4 = tf.layers.dropout(inputs=h4, rate=self.conv_prob, name='conv4_drop')
            #h5 = lrelu((conv3d(input_=h4, out_channels=256, k_d=1, k_h=1, k_w=1, s_d=1, s_h=1, s_w=1, name='h5_conv3d')), 0.2)
         
            l1 = tf.contrib.layers.flatten(h4)
            l1 = tf.layers.dropout(inputs=l1, rate=self.fc_prob, name="adv_dropout")
            #l1 = lrelu(batch_norm(name='bn6')linear(l1, 128, scope="Linear1024"), 0.2)
            out = linear(l1, 1, scope="final_decision")
            return out

            #print("b4 3d", self.input.get_shape())
            #print("after 3d:", h5.get_shape())
            #sys.exit()
            
