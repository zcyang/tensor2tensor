"""Attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import tensorflow.contrib.slim as slim

# pylint: disable=unused-import
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_152
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152


@registry.register_model
class VqaAttentionBaseline(t2t_model.T2TModel):
  """Attention baseline model for VQA."""

  def body(self, features):
    hp = self.hparams
    image_feat = image_embedding(
        features["inputs"],
        trainable=hp.train_resnet,
        is_training=hp.mode == tf.estimator.ModeKeys.TRAIN)

    # apply layer normalization and dropout on image_feature
    image_feat = common_layers.postprocess(image_feat, hp)
    # apply dropout on question embedding
    question = tf.nn.dropout(features["question"], 1. - hp.dropout)

    query = question_encoder(question, hp)
    image_ave = attn(image_feat, query, hp)

    image_question = tf.concatenate([image_ave, query], axis=1)
    image_question = tf.nn.dropout(image_question, 1. - hp.dropout)

    output = mlp(image_question, hp)

    # Expand dimension 1 and 2
    return tf.expand_dims(tf.expand_dims(output, axis=1), axis=2)

  def infer(self,
            features,
            decode_length=1,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """Predict."""
    del decode_length, beam_size, top_beams, alpha, use_tpu
    assert features is not None
    logits, _ = self(features)
    assert len(logits.get_shape()) == 5
    logits = tf.squeeze(logits, [1, 2, 3])
    log_probs = common_layers.log_prob_from_logits(logits)
    predictions, scores = common_layers.argmax_with_score(log_probs)
    return {
        "outputs": predictions,
        "scores": scores,
    }


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
  """Extract image features from pretrained resnet model."""

  is_resnet_training = trainable and is_training

  batch_norm_params = {
      "is_training": is_resnet_training,
      "trainable": trainable,
      "decay": batch_norm_decay,
      "epsilon": batch_norm_epsilon,
      "scale": batch_norm_scale,
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
        with slim.arg_scope([slim.max_pool2d], padding="SAME"):
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
      rnn_cell(hparams.hidden_size),
      output_keep_prob=1.0 - hparams.dropout)


def question_encoder(question, hparams, name="encoder"):
  """Question encoder, run LSTM encoder and get the last output as encoding."""
  with tf.variable_scope(name, "encoder", values=[question]):
    question = common_layers.flatten4d3d(question)
    padding = common_attention.embedding_to_padding(question)
    length = common_attention.padding_to_length(padding)

    rnn_layers = [_get_rnn_cell(hparams)
                  for _ in range(hparams.num_rnn_layers)]
    rnn_multi_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
    outputs, _ = tf.nn.dynamic_rnn(rnn_multi_cell, question, length)

    batch_size = common_layers.shape_list(outputs)[0]
    row_indices = tf.range(batch_size)
    indices = tf.transpose([row_indices, length])
    last_output = tf.gather_nd(outputs, indices)

  return last_output


def attn(image_feat, query, hparams, name="attn"):
  """Attention on image feature with question as query."""
  with tf.variable_scope(name, "attn", values=[image_feat, query]):
    attn_dim = hparams.attn_dim
    num_glimps = hparams.num_glimps
    batch_size, _, _, num_channels = image_feat.get_shape().as_list()
    image_feat = common_layers.flatten4d3d(image_feat)
    query = tf.expand_dims(query, 1)
    image_proj = common_attention.compute_attention_component(
        image_feat, attn_dim, name="image_proj")
    query_proj = common_attention.compute_attention_component(
        query, attn_dim, name="query_proj")
    h = tf.nn.relu(image_proj + query_proj)
    h_proj = common_attention.compute_attention_component(
        h, num_glimps, name="h_proj")
    p = tf.nn.softmax(h_proj, dim=1)
    image_ave = tf.matmul(image_feat, p, transpose_a=True)
    image_ave = tf.reshape(image_ave, [batch_size, num_channels*num_glimps])

    return image_ave


def mlp(feature, hparams, name="mlp"):
  """Multi layer perceptron with dropout and relu activation."""
  with tf.variable_scope(name, "mlp", values=[feature]):
    num_mlp_layer = hparams.num_mlp_layer
    mlp_dim = hparams.mlp_dim
    for i in range(num_mlp_layer):
      feature = common_layers.dense(feature, mlp_dim, activation=tf.relu)
      feature = common_layers.dropout_with_broadcast_dims(
          feature, keep_prob=1-hparams.dropout, name="layer_%i"%(i))
    return feature


@registry.register_hparams
def vqa_attention_base():
  """VQA attention baseline hparams."""
  hparams = common_hparams.basic_hparams1()
  hparams.dropout = 0.5
  hparams.norm_type = "layer"
  hparams.layer_postprocess_sequence = "nd"
  hparams.layer_prepostprocess_dropout = 0.5

  # add new hparams
  hparams.add_hparams("train_resnet", False)
  hparams.add_hparams("rnn_type", "lstm_layernorm")
  hparams.add_hparams("num_rnn_layers", 2)

  hparams.add_hparams("attn_dim", 512)
  hparams.add_hparams("num_glimps", 2)

  hparams.add_hparams("num_mlp_layers", 1)
  hparams.add_hparams("mlp_dim", 1024)

  return hparams
