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

import models
import tensorflow as tf
from tensorflow import flags
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib

FLAGS = flags.FLAGS
local_device_protos = device_lib.list_local_devices()
gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
# gpus = gpus[:FLAGS.num_gpu]
num_gpus = len(gpus)


class inference_lateFuse_ConvTemporalGRU(models.BaseModel):
    # Baseline1: Use 2-layer MLP to predict flow based on rgb input, then concate two to MLP video classifier
    def create_model(self, rgb_input, vocab_size, mask, l2_penalty=1e-8, **unused_params):
        last_relu = FLAGS.last_relu
        gru_layers = FLAGS.gru_layers
        rgb_feature_size = rgb_input.get_shape().as_list()[2]
        rgb_feature = rgb_input
        feature_num = rgb_feature.get_shape().as_list()[1]
        rgb_feature = tf.reshape(rgb_feature, [-1, rgb_feature_size])
        dropout_keep_prob = 1.0

        rgb_feature = tf.nn.dropout(rgb_feature, dropout_keep_prob)
        rgb_feature = tf.reshape(rgb_feature, [-1, feature_num, rgb_feature_size])

        s_output = tf.zeros([tf.shape(rgb_input)[0], feature_num, rgb_feature_size], name='s_hidden')
        sm1_pad = tf.zeros([tf.shape(rgb_input)[0], 1, rgb_feature_size], name='sm1_pad')
        sp1_pad = tf.zeros([tf.shape(rgb_input)[0], 1, rgb_feature_size], name='sp1_pad')

        with tf.variable_scope("predictor"):
            with tf.variable_scope('cnn_trans_x'):
                x_hidden = tf.layers.conv1d(inputs=rgb_feature * mask, filters=6 * rgb_feature_size, kernel_size=1, strides=1, padding='same', activation=None)
                z_tx, z_tp1x, z_tm1x, r_tp1x, r_tm1x, h_tx = tf.split(x_hidden, 6, axis=-1)  # z_{t,x}, z_{t+1,x}, z_{t-1,x},  r_{t+1,x}, r_{t-1,x}, z_{t,x}

            for i in range(gru_layers):
                with tf.variable_scope('cnn_trans', reuse=True if i > 0 else None):
                    s_hidden = tf.layers.conv1d(inputs=s_output, filters=5 * rgb_feature_size, kernel_size=3, strides=1, padding='same', activation=None)
                    z_ts, z_tp1s, z_tm1s, r_tp1s, r_tm1s = tf.split(s_hidden, 5, axis=-1)
                    # z_ts, z_tp1s, z_tm1s, r_tp1s, r_tm1s = tf.squeeze(z_ts, axis=-1), tf.squeeze(z_tp1s, axis=-1), tf.squeeze(z_tm1s, axis=-1), tf.squeeze(r_tp1s, axis=-1), tf.squeeze(r_tm1s, axis=-1)

                    r_tp1 = tf.nn.sigmoid(r_tp1x + r_tp1s)
                    r_tm1 = tf.nn.sigmoid(r_tm1x + r_tm1s)
                    z_t_pi = z_tx + z_ts
                    z_tp1_pi = z_tp1x + z_tp1s
                    z_tm1_pi = z_tm1x + z_tm1s
                    z_t, z_tp1, z_tm1 = tf.split(tf.nn.softmax(tf.stack([z_t_pi, z_tp1_pi, z_tm1_pi], axis=3), axis=3), 3, axis=3)
                    z_t, z_tp1, z_tm1 = tf.squeeze(z_t, axis=3), tf.squeeze(z_tp1, axis=3), tf.squeeze(z_tm1, axis=3)

                    s_tm1 = tf.concat([sm1_pad, s_output[:, :-1, :]], 1)
                    s_tp1 = tf.concat([s_output[:, 1:, :], sp1_pad], 1)
                    h_t_stm1 = slim.fully_connected(s_tm1 * r_tm1, rgb_feature_size, activation_fn=None)
                    h_t_stp1 = slim.fully_connected(s_tp1 * r_tp1, rgb_feature_size, activation_fn=None)

                    h_t = h_tx + h_t_stm1 + h_t_stp1
                    if last_relu:
                        h_t = tf.nn.relu(h_t)
                    else:
                        h_t = tf.nn.tanh(h_t)

                    s_output = z_t * h_t + z_tp1 * s_tp1 + z_tm1 * s_tm1

            predicted_flow = s_output

            predicted_flow_reshape = tf.reshape(predicted_flow, [-1, feature_num, rgb_feature_size])
            predicted_flow = tf.reshape(predicted_flow, [-1, rgb_feature_size])

        return predicted_flow_reshape
