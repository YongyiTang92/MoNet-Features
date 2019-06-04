# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains the base class for models."""
import tensorflow as tf
import i3d_modified


class BaseModel(object):
    """Inherit from this class when implementing new models."""

    def create_model(self, unused_model_input, **unused_params):
        raise NotImplementedError()


class rgb_I3D(object):

    def create_model(self, video_clip, vocab_size, channels=3, is_training=False, dropout_keep_prob=0.5, **unused_params):
        if not is_training:
            dropout_keep_prob = 1.0
        video_reshape = tf.transpose(video_clip, [0, 1, 3, 4, 2])  # (batch, frames, height, width, channels)
        with tf.variable_scope('RGB'):
            with tf.variable_scope('inception_i3d'):
                i3d_model = i3d_modified.InceptionI3d(num_classes=vocab_size, final_endpoint='Logits', name='inception_i3d')
                logits, end_points = i3d_model._build(video_reshape, is_training=is_training, dropout_keep_prob=dropout_keep_prob)
        logits = end_points["Logits"]
        logits = tf.reduce_mean(logits, axis=1)
        feature = end_points['feature_before_avg']
        return {"logits": logits, "feature": feature}

    def get_exclusive_scope(self):
        exclusive_scope = ['AuxLogits/', 'Logits/']
        return exclusive_scope
