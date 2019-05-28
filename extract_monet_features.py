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

import tensorflow as tf
import numpy as np
from frame_level_models import inference_lateFuse_ConvTemporalGRU
from tensorflow import flags
from utils import SaverRestore
from models import rgb_I3D

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "last_relu", False,
    "Whether adding ReLU in the last layer")
flags.DEFINE_integer("gru_layers", 5, "Number of GRU layers.")
tf.reset_default_graph()


def main():
    rgb_video = tf.placeholder(tf.float32, shape=[1, 64, 3, 224, 224])  # video segment with 64 frames

    i3d_result = rgb_I3D().create_model(rgb_video, vocab_size=400)
    i3d_features = i3d_result['feature']
    print(i3d_features)
    mask = tf.ones([1, i3d_features.get_shape()[1], 1])
    monet_features = inference_lateFuse_ConvTemporalGRU().create_model(tf.squeeze(i3d_features, [2, 3]), vocab_size=400, mask=mask)
    # case1 normal save and restore
    # define simple graphs
    with tf.Session() as sess:
        saver = SaverRestore('./Monet_model.ckpt')
        saver._setup_graph()
        saver._run_init(sess)

        feed_dict = {rgb_video: np.ones([1, 64, 3, 224, 224], dtype=np.float32)}
        monet_features_ = sess.run(monet_features, feed_dict=feed_dict)
        print(monet_features_.shape)


if __name__ == "__main__":
    main()
