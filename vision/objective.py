# coding=utf-8
# Copyright 2020.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

FLAGS = flags.FLAGS

LARGE_NUM = 1e9

def add_gradients_penalty(x, model, model_train_mode):
  """https://colab.research.google.com/github/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb#scrollTo=Wyipg-4oSYb1"""
  with tf.GradientTape() as t:
    t.watch(x)
    hidden = model(x, is_training=model_train_mode)
  gradients = t.gradient(hidden, x)
  dx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3]))
  d_regularizer = tf.reduce_mean((dx - 1.0) ** 2)
  return d_regularizer

def add_supervised_loss(labels, logits, weights, **kwargs):
  """Compute loss for model and add it to loss collection."""
  return tf.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0,
                         loss_type=None,
                         flags=None):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf.split(hidden, 2, 0)
  batch_size = tf.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if tpu_context is not None:
    hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
    hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)
  else:
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
  # check WPC
  if loss_type.lower() == "wpc":
    assert flags.gradient_penalty_weight != 0.0
  else:
    assert flags.gradient_penalty_weight == 0.0
  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  # aa and bb diagonals are not positive samples; positive samples are ab abd ba diagnoals
  # thus we want to mask aa and bb diagonals out
  if loss_type.lower() == "nce" or loss_type.lower() == "dv" or loss_type.lower() == "wpc":
    # NCE loss: minus big number to create cloes to 0 values in softmax
    print(loss_type)
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = logits_bb - masks * LARGE_NUM
  else: # otherwise just mask out using 0
    logits_aa = logits_aa * (1 - masks)
    logits_bb = logits_bb * (1 - masks)
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
  #############################################################################
  ### Different losses: nce, chi, js and nwj
  ### Pos_scores: positive samples, i.e. joint distribution terms
  ### neg_scores: negative samples, i.e. marginal distribution terms
  #############################################################################
  if loss_type.lower() == "nce":
    loss_a = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.losses.softmax_cross_entropy(
        labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
  elif loss_type.lower() == "chi":
    ## Chi squared loss in general form
    alpha = flags.alpha
    beta = flags.beta
    gamma = flags.gamma

    joint_a = labels * tf.concat([logits_ab, logits_aa], 1)
    joint_b = labels * tf.concat([logits_ba, logits_bb], 1)
    # non-correlated views
    marg_a  = (1.-labels) * tf.concat([logits_ab, logits_aa], 1)
    marg_b  = (1.-labels) * tf.concat([logits_ba, logits_bb], 1)
    batch_size = tf.cast(batch_size, tf.float32)
    joint = 0.5*(tf.reduce_sum(joint_a - 0.5 * beta * joint_a**2) /  batch_size)\
            + 0.5*(tf.reduce_sum(joint_b - 0.5 * beta * joint_b**2) /  batch_size)
    # non-correlated views
    marg = 0.5*(tf.reduce_sum(alpha * marg_a + 0.5 * gamma * marg_a**2) /  (2*batch_size*(batch_size-1.)))\
            + 0.5*(tf.reduce_sum(alpha * marg_b + 0.5 * gamma * marg_b**2) /  (2*batch_size*(batch_size-1.)))
    loss = -1. * (joint - marg)
    tf.losses.add_loss(loss)
    return loss, logits_ab, labels

  elif loss_type.lower() == "js":
    # Jensen Shannon 
    def js(logits_concat, labels, scope=None):
      lbls = math_ops.cast(labels, logits_concat.dtype)
      """SHOULD I ADD STOP GRADIENT?"""
      bs = math_ops.cast(batch_size, logits_concat.dtype)
      pos_scores = tf.reduce_sum(lbls * (-tf.math.softplus(-logits_concat))) / bs
      neg_scores = tf.reduce_sum((1 - lbls) * tf.math.softplus(logits_concat)) / ((2 * bs - 1) * bs)
      return - (pos_scores - neg_scores)
      
    loss_a = 0.5 * js(tf.concat([logits_ab, logits_aa], 1), labels)
    loss_b = 0.5 * js(tf.concat([logits_ba, logits_bb], 1), labels)
    tf.losses.add_loss(loss_a)
    tf.losses.add_loss(loss_b)

  elif loss_type.lower() == "nwj":
    def nwj(logits_concat, labels, scope=None):
      lbls = math_ops.cast(labels, logits_concat.dtype)
      """SHOULD I ADD STOP GRADIENT?"""
      bs = math_ops.cast(batch_size, logits_concat.dtype)
      pos_scores = tf.reduce_sum(lbls * logits_concat) / bs
      neg_scores = tf.reduce_sum((1 - lbls) * tf.math.exp(logits_concat - 1)) / ((2 * bs - 1) * bs)
      return - (pos_scores - neg_scores)

    loss_a = 0.5 * nwj(tf.concat([logits_ab, logits_aa], 1), labels)
    loss_b = 0.5 * nwj(tf.concat([logits_ba, logits_bb], 1), labels) 
    tf.losses.add_loss(loss_a)
    tf.losses.add_loss(loss_b)
  elif loss_type.lower() == "dv":
    # Donsker and Varadhan 
    def dv(logits_concat, labels, scope=None):
      lbls = math_ops.cast(labels, logits_concat.dtype)
      """SHOULD I ADD STOP GRADIENT?"""
      bs = math_ops.cast(batch_size, logits_concat.dtype)
      pos_scores = tf.reduce_sum(lbls * logits_concat) / bs
      neg_scores = tf.math.reduce_logsumexp((1 - lbls) * logits_concat) - tf.math.log((2 * bs - 1) * bs)
      return - (pos_scores - neg_scores)

    loss_a = 0.5 * dv(tf.concat([logits_ab, logits_aa], 1), labels)
    loss_b = 0.5 * dv(tf.concat([logits_ba, logits_bb], 1), labels) 
    tf.losses.add_loss(loss_a)
    tf.losses.add_loss(loss_b)
  elif loss_type.lower() == "wpc":
    # Wasserstein Dependency Measure (i.e. Wasserstein Predictive Coding)
    # Adding soon
    pass # operation performed in model.py
    
  else:
    raise ValueError("Loss not implemented yet; only support {nce, chi, js, nwj}")
    
  loss = loss_a + loss_b
  return loss, logits_ab, labels


def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
