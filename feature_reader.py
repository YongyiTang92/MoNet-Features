#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import tensorflow as tf
from tensorflow import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS
tf.flags.DEFINE_string("data_dir", './train_features',
                       "Directory to read tfrecoreds from")


class Data_loader(object):
    """docstring for ClassName"""

    def __init__(self, filenames_, max_frames=32, num_classes=400, feature_size=1024):
        self.feature_keys = ['rgb_feature', 'flow_feature', 'Monet_flow_feature']
        self.feature_size = feature_size
        self.num_classes = num_classes
        self.max_frames = max_frames
        dataset = tf.data.TFRecordDataset(filenames_)
        dataset = dataset.map(self.parser_fn, num_parallel_calls=16)
        dataset = dataset.repeat(1)
        dataset = dataset.batch(1)
        self.dataset = dataset.prefetch(2)

    def parser_fn(self, example_proto):
        """
        Parse tf example object to tensor

        Returns:
            rgb_feature, flow_feature, labels, num_feature: if flow_image is None
            else return rgb_feature, flow_feature, flow_x, flow_y, labels, num_feature, num_frames

        """
        contexts, features = tf.parse_single_sequence_example(
            example_proto,
            context_features={"num_frames": tf.FixedLenFeature(  # number of images
                [], tf.int64),
                "num_feature": tf.FixedLenFeature(  # number of feature in a video
                [], tf.int64),
                "label_index": tf.FixedLenFeature(
                [], tf.int64),
                "label_name": tf.FixedLenFeature(
                [], tf.string),
                "video": tf.FixedLenFeature(
                [], tf.string)},
            sequence_features={
                feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
                for feature_name in self.feature_keys
            })
        # read ground truth labels
        labels = (tf.cast(tf.sparse_to_dense(contexts["label_index"], (self.num_classes,), 1,
                                             validate_indices=False), tf.int32))

        num_feature_type = len(self.feature_keys)
        feature_matrices = [None] * num_feature_type  # an array of different features
        num_feature = -1
        for feature_index in range(num_feature_type):
            feature_matrix, num_frames_in_this_feature = self.get_video_matrix(features[self.feature_keys[feature_index]], subsample=False)
            feature_matrices[feature_index] = feature_matrix
            if num_feature == -1:
                num_feature = num_frames_in_this_feature

        rgb_feature = feature_matrices[0]
        flow_feature = feature_matrices[1]
        monet_flow_feature = feature_matrices[2]
        num_feature = tf.minimum(num_feature, self.max_frames)

        index_ones = tf.ones([num_feature, 1])
        index_zeros = tf.zeros([self.max_frames - num_feature, 1])
        feature_mask = tf.concat([index_ones, index_zeros], 0)

        return rgb_feature, flow_feature, monet_flow_feature, labels, num_feature, feature_mask, contexts

    def get_video_matrix(self, features, subsample=True):
        """Decodes features from an input string and quantizes it.

        Args:
          features: raw feature values
          max_frames: number of frames (rows) in the output feature_matrix

        Returns:
          feature_matrix: matrix of all frame-features
          num_frames: number of frames in the sequence
        """
        if subsample:
            decoded_features = tf.reshape(tf.decode_raw(features, tf.float32),
                                          [-1, self.feature_size])
            decoded_features = decoded_features[::8, :]
        else:
            decoded_features = tf.reshape(tf.decode_raw(features, tf.float32),
                                          [-1, self.feature_size])

        num_frames = tf.minimum(tf.shape(decoded_features)[0], self.max_frames)
        feature_matrix = self.repeat_last(decoded_features, 0, self.max_frames)
        return feature_matrix, num_frames

    def repeat_last(self, tensor, axis, new_size):
        """Truncates or pads a tensor to new_size on on a given axis.

        Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
        size increases, the padding will be performed at the end, using the last feature.

        Args:
          tensor: The tensor to be resized.
          axis: An integer representing the dimension to be sliced.
          new_size: An integer or 0d tensor representing the new value for
            tensor.shape[axis].

        Returns:
          The resized tensor.
        """
        tensor = tf.convert_to_tensor(tensor)
        shape = tf.unstack(tf.shape(tensor))

        pad_shape = shape[:]
        pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

        shape[axis] = tf.minimum(shape[axis], new_size)
        shape = tf.stack(shape)

        tensor_reshape = tf.reshape(tensor, [-1, self.feature_size])
        repeat_last = tf.tile(tensor_reshape[-1:, :], [pad_shape[axis], 1])
        # repeat_last = tf.reshape(repeat_last, [-1])

        resized = tf.concat([
            tf.slice(tensor, tf.zeros_like(shape), shape),
            repeat_last
        ], axis)

        # Update shape.
        new_shape = tensor.get_shape().as_list()  # A copy is being made.
        new_shape[axis] = new_size
        resized.set_shape(new_shape)
        return resized


def main():
    tf.reset_default_graph()
    tfrecords_path = gfile.Glob(os.path.join(FLAGS.data_dir, "*.tfrecords"))
    tfrecords_path.sort()
    # prepare tfrecord
    max_frames = 32
    num_classes = 400
    feature_size = 1024
    with tf.device('cpu:0'):
        filenames_ = tfrecords_path
        data_loader = Data_loader(filenames_, max_frames, num_classes, feature_size)
        iterator = data_loader.dataset.make_one_shot_iterator()
        rgb_feature, flow_feature, monet_flow_feature, labels, num_feature_tensor, feature_mask, contexts = iterator.get_next()
    # case1 normal save and restore
    # define simple graphs
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while True:
                rgb_feature_, flow_feature_, monet_flow_feature_, label_, num_feature_, feature_mask_, contexts_ = sess.run([rgb_feature, flow_feature, monet_flow_feature, labels, num_feature_tensor, feature_mask, contexts])
        except tf.errors.OutOfRangeError, e:
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    main()
