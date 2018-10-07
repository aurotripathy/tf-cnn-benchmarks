# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""P1B3 Model

"""

import tensorflow as tf
from models import model


class P1B3Model(model.MLPModel):

    def __init__(self, params=None):
        super(P1B3Model, self).__init__(
            # model-name, feature-size, batch-size, learning-rate
            'p1b3net', 32, 128, 0.1, params=params)

    def add_inference(self, cnn):
        cnn.affine(100, stddev=0.04, bias=0.1)
        cnn.affine(50, stddev=0.04, bias=0.1)

    def get_learning_rate(self, global_step, batch_size):
        num_examples_per_epoch = 50000  # ToDO
        num_epochs_per_decay = 100  # ToDO
        decay_steps = int(
            num_epochs_per_decay * num_examples_per_epoch / batch_size)
        decay_factor = 0.1
        return tf.train.exponential_decay(
            self.learning_rate,
            global_step,
            decay_steps,
            decay_factor,
            staircase=True)
