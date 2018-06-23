"""Attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_152
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152

class AttentionBaseline(t2t_model.T2TModel):
  def __init__(self, *args, **kwargs):
    super(AttentionBaseline, self).__init__(*args, **kwargs)


  def body(self, features):
    hp = self.hparams


def image_embedding(images,
                    model_fn=resnet_v2_152,
                    trainable=True,
                    is_training=True,
                    weight_decay=0.0001,
                    batch_norm_decay=0.997,
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    add_summaries=False,
                    reuse=False):

  is_resnet_training = trainable and is_training

  batch_norm_params = {
    'is_training': is_resnet_training,
    'trainable': trainable,
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
  }

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(model_fn.__name__, [images], reuse=reuse) as scope:
    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=slim.variance_scaling_initializer(),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
          net, end_points = model_fn(
              images, num_classes=None, global_pool=False,
              is_training=is_resnet_training,
              reuse=reuse, scope=scope)

  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net

def _get_rnn_cell(hparams):
  if hparams.rnn_type == "lstm":
    rnn_cell = tf.contrib.rnn.BasicLSTMCell
  elif hparams.rnn_type == "lstm_layernorm":
    rnn_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
  return tf.contrib.rnn.DropoutWrapper(
      rnn_cell(hparmas.hidden_size),
      output_keep_prob=1.0 - hparams.rnn_output_dropout)

def question_encoder(question, hparams, name="encoder"):
  """Question encoder, run LSTM encoder and get the last output as encoding."""
  question = common_layers.flatten4d3d(question)
  padding = common_attention.embedding_to_padding(question)
  length = common_attention.padding_to_length(padding)

  rnn_layer = [ _get_rnn_cell(hparams)
                for _ in range(hparams.num_rnn_layers)]
  rnn_multi_cell = tf.contrib.rnn.MultiRNNCell(layers)
  outputs, state = tf.nn.dynamic_rnn(rnn_multi_cell, question, length)

  batch_size = common_layers.shape_list(x)[0]
  row_indices = tf.range(batch_size)
  indices =  tf.transpose([row_indices, length])
  last_output = tf.gather_nd(outputs, indices)

  return last_output


def vqa_attention_base():
  hparams = common_hparams.basic_hparams1()

  # add new hparams
  hparmas.add_hparams("rnn_type", "lstm_layernorm")
  hparmas.add_hparams("rnn_output_dropout", 0.5)
  hparmas.add_hparams("num_rnn_layers", 2)
