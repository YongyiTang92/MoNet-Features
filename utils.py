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

"""Contains a collection of util functions for training and evaluating.
"""

import numpy as np
import tensorflow as tf
from tensorflow import logging
import os
import six

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

    Args:
        feat_vector: the input 1-d vector.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.

    Returns:
        A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
    """Add the global_step summary to the Tensorboard.

    Args:
        summary_writer: Tensorflow summary_writer.
        global_step_val: a int value of the global step.
        global_step_info_dict: a dictionary of the evaluation metrics calculated for
            a mini-batch.
        summary_scope: Train or Eval.

    Returns:
        A string of this global_step summary
    """
    acc_1 = global_step_info_dict["accuracy@1"]
    acc_5 = global_step_info_dict["accuracy@5"]
    this_loss = global_step_info_dict["loss"]
    examples_per_second = global_step_info_dict.get("examples_per_second", -1)

    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Acc@1", acc_1),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Acc@5", acc_5),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
        global_step_val)

    if examples_per_second != -1:
        summary_writer.add_summary(
            MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                        examples_per_second), global_step_val)

    summary_writer.flush()
    info = ("global_step {0} | Batch_Acc@1: {1:.3f} | Batch_Acc@5: {2:.3f} | Batch Loss: {3:.3f} "
            "| Examples_per_sec: {4:.3f}").format(
        global_step_val, acc_1, acc_5, this_loss,
        examples_per_second)
    return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
    """Add the epoch summary to the Tensorboard.

    Args:
        summary_writer: Tensorflow summary_writer.
        global_step_val: a int value of the global step.
        epoch_info_dict: a dictionary of the evaluation metrics calculated for the
            whole epoch.
        summary_scope: Train or Eval.

    Returns:
        A string of this global_step summary
    """
    epoch_id = epoch_info_dict["epoch_id"]
    avg_acc_1 = epoch_info_dict["avg_accuracy@1"]
    avg_acc_5 = epoch_info_dict["avg_accuracy@5"]
    avg_loss = epoch_info_dict["avg_loss"]
    rgb_avg_acc_1 = epoch_info_dict["RGB_AvgAccuracy@1"]
    flow_avg_acc_1 = epoch_info_dict["Flow_AvgAccuracy@1"]
    gtflow_avg_acc_1 = epoch_info_dict["gtFlow_AvgAccuracy@1"]
    feature_distance = epoch_info_dict["feature_distance"]
    kl_loss = epoch_info_dict["kl_loss"]
    D_loss = epoch_info_dict["D_loss"]
    G_loss = epoch_info_dict["G_loss"]
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Acc@1", avg_acc_1),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Acc@5", avg_acc_5),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_RGBAvg_Acc@1", rgb_avg_acc_1),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_FlowAvg_Acc@1", flow_avg_acc_1),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_gtFlowAvg_Acc@1", gtflow_avg_acc_1),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_feature_distance", feature_distance),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_kl_loss", kl_loss),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_D_loss", D_loss),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_G_loss", G_loss),
        global_step_val)
    summary_writer.flush()

    info = ("epoch/eval number {0} | Avg_Acc@1: {1:.4f} | Avg_Acc@5: {2:.4f} "
            "| Avg_Loss: {3:4f} | predict_flow_Acc@1: {4:.4f} | rgbw_Acc@1: {5:.4f}").format(epoch_id, avg_acc_1, avg_acc_5, avg_loss, flow_avg_acc_1, rgb_avg_acc_1)
    return info


def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
    """Extract the list of feature names and the dimensionality of each feature
         from string of comma separated values.

    Args:
        feature_names: string containing comma separated list of feature names
        feature_sizes: string containing comma separated list of feature sizes

    Returns:
        List of the feature names and list of the dimensionality of each feature.
        Elements in the first/second list are strings/integers.
    """
    list_of_feature_names = [
        feature_names.strip() for feature_names in feature_names.split(',')]
    list_of_feature_sizes = [
        int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
    if len(list_of_feature_names) != len(list_of_feature_sizes):
        logging.error("length of the feature names (=" +
                      str(len(list_of_feature_names)) + ") != length of feature "
                      "sizes (=" + str(len(list_of_feature_sizes)) + ")")

    return list_of_feature_names, list_of_feature_sizes


def clip_gradient_norms(gradients_to_variables, max_norm):
    """Clips the gradients by the given value.

    Args:
        gradients_to_variables: A list of gradient to variable pairs (tuples).
        max_norm: the maximum norm value.

    Returns:
        A list of clipped gradient to variable pairs.
    """
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars


def combine_gradients(tower_grads):
    """Calculate the combined gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been summed
         across all towers.
    """
    filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
    final_grads = []
    for i in xrange(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in xrange(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((grad, filtered_grads[0][i][1],))

    return final_grads


def combine_gradients_without_scope(tower_grads, scope='tower/classifier'):
    """Calculate the combined gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been summed
         across all towers.
    """
    # filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
    filtered_grads = []
    for grad_list in tower_grads:
        tmp_filtered_grads = []
        for x in grad_list:
            if (x[0] is not None) and (x[1] not in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)):
                tmp_filtered_grads.append(x)
        filtered_grads.append(tmp_filtered_grads)

    final_grads = []
    for i in xrange(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in xrange(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((grad, filtered_grads[0][i][1],))

    return final_grads


def combine_gradients_with_scope(tower_grads, scope='tower/classifier'):
    """Calculate the combined gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been summed
         across all towers.
    """
    # filtered_grads = [[x for x in grad_list if x[0] is not None] for grad_list in tower_grads]
    filtered_grads = []
    for grad_list in tower_grads:
        tmp_filtered_grads = []
        for x in grad_list:
            if (x[0] is not None) and (x[1] in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)):
                tmp_filtered_grads.append(x)
        filtered_grads.append(tmp_filtered_grads)

    final_grads = []
    for i in xrange(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in xrange(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((grad, filtered_grads[0][i][1],))

    return final_grads


def read_checkpoint_vars(model_path):
    """ return a set of strings """
    reader = tf.train.NewCheckpointReader(model_path)
    reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
    ckpt_vars = reader.get_variable_to_shape_map().keys()
    return reader, set(ckpt_vars)


class CheckpointReaderAdapter(object):
    """
    An adapter to work around old checkpoint format, where the keys are op
    names instead of tensor names (with :0).
    """

    def __init__(self, reader):
        self._reader = reader
        m = self._reader.get_variable_to_shape_map()
        self._map = {k if k.endswith(':0') else k + ':0': v
                     for k, v in six.iteritems(m)}

    def get_variable_to_shape_map(self):
        return self._map

    def get_tensor(self, name):
        if self._reader.has_tensor(name):
            return self._reader.get_tensor(name)
        if name in self._map:
            assert name.endswith(':0'), name
            name = name[:-2]
        return self._reader.get_tensor(name)

    def has_tensor(self, name):
        return name in self._map

    # some checkpoint might not have ':0'
    def get_real_name(self, name):
        if self._reader.has_tensor(name):
            return name
        assert self.has_tensor(name)
        return name[:-2]


def get_savename_from_varname(
        varname, varname_prefix=None,
        savename_prefix=None):
    """
    Args:
        varname(str): a variable name in the graph
        varname_prefix(str): an optional prefix that may need to be removed in varname
        savename_prefix(str): an optional prefix to append to all savename
    Returns:
        str: the name used to save the variable
    """
    name = varname
    if varname_prefix is not None \
            and name.startswith(varname_prefix):
        name = name[len(varname_prefix) + 1:]
    if savename_prefix is not None:
        name = savename_prefix + '/' + name
    return name


class SessionInit(object):
    """ Base class for utilities to load variables to a (existing) session. """

    def init(self, sess):
        """
        Initialize a session

        Args:
            sess (tf.Session): the session
        """
        self._setup_graph()
        self._run_init(sess)

    def _setup_graph(self):
        pass

    def _run_init(self, sess):
        pass


class SaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """

    def __init__(self, model_path, prefix=None, ignore=[]):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint.
            ignore (list[str]): list of tensor names that should be ignored during loading, e.g. learning-rate
        """
        model_path = get_checkpoint_path(model_path)
        self.path = model_path  # attribute used by AutoResumeTrainConfig!
        self.prefix = prefix
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]

    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def _run_init(self, sess):
        print("Restoring checkpoint from {} ...".format(self.path))
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()

        mismatch = MismatchLogger('graph', 'checkpoint')
        for v in graph_vars:
            name = get_savename_from_varname(v.name, savename_prefix=self.prefix)
            if name in self.ignore and reader.has_tensor(name):
                print("Variable {} in the graph will not be loaded from the checkpoint!".format(name))
            else:
                if reader.has_tensor(name):
                    func(reader, name, v)
                    chkpt_vars_used.add(name)
                else:
                    # use tensor name (instead of op name) for logging, to be consistent with the reverse case
                    if not is_training_name(v.name):
                        mismatch.add(v.name)
        mismatch.log()
        mismatch = MismatchLogger('checkpoint', 'graph')
        if len(chkpt_vars_used) < len(chkpt_vars):
            unused = chkpt_vars - chkpt_vars_used
            for name in sorted(unused):
                if not is_training_name(name):
                    mismatch.add(name)
        mismatch.log()

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict


def get_checkpoint_path(model_path):
    """
    Work around TF problems in checkpoint path handling.

    Args:
        model_path: a user-input path
    Returns:
        str: the argument that can be passed to NewCheckpointReader
    """
    if os.path.basename(model_path) == model_path:
        model_path = os.path.join('.', model_path)  # avoid #4921 and #6142
    if os.path.basename(model_path) == 'checkpoint':
        assert tf.gfile.Exists(model_path), model_path
        model_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = model_path
    if '00000-of-00001' in model_path:
        new_path = model_path.split('.data')[0]
    elif model_path.endswith('.index'):
        new_path = model_path.split('.index')[0]
    if new_path != model_path:
        print(
            "Checkpoint path {} is auto-corrected to {}.".format(model_path, new_path))
        model_path = new_path
    assert tf.gfile.Exists(model_path) or tf.gfile.Exists(model_path + '.index'), model_path
    return model_path


def is_training_name(name):
    """
    **Guess** if this variable is only used in training.
    Only used internally to avoid too many logging. Do not use it.
    """
    # TODO: maybe simply check against TRAINABLE_VARIABLES and MODEL_VARIABLES?
    # TODO or use get_slot_names()
    name = get_op_tensor_name(name)[0]
    if name.endswith('/Adam') or name.endswith('/Adam_1'):
        return True
    if name.endswith('/Momentum'):
        return True
    if name.endswith('/Adadelta') or name.endswith('/Adadelta_1'):
        return True
    if name.endswith('/RMSProp') or name.endswith('/RMSProp_1'):
        return True
    if name.endswith('/Adagrad'):
        return True
    if name.startswith('EMA/'):  # all the moving average summaries
        return True
    if name.startswith('AccumGrad') or name.endswith('/AccumGrad'):
        return True
    if name.startswith('apply_gradients'):
        return True
    return False


def get_op_tensor_name(name):
    """
    Will automatically determine if ``name`` is a tensor name (ends with ':x')
    or a op name.
    If it is an op name, the corresponding tensor name is assumed to be ``op_name + ':0'``.

    Args:
        name(str): name of an op or a tensor
    Returns:
        tuple: (op_name, tensor_name)
    """
    if len(name) >= 3 and name[-2] == ':':
        return name[:-2], name
    else:
        return name, name + ':0'


class MismatchLogger(object):
    def __init__(self, exists, nonexists):
        self._exists = exists
        self._nonexists = nonexists
        self._names = []

    def add(self, name):
        self._names.append(get_op_tensor_name(name)[0])

    def log(self):
        if len(self._names):
            print("The following variables are in the {}, but not found in the {}: {}".format(
                self._exists, self._nonexists, ', '.join(self._names)))
