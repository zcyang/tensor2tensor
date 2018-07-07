"""Attention models for VQA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import restore_hook
from tensor2tensor.utils import t2t_model

import tensorflow as tf

# pylint: disable=unused-import
from google3.third_party.tensorflow.contrib import slim
from google3.third_party.tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_152
from google3.third_party.tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152


@registry.register_model
class VqaAttentionBaseline(t2t_model.T2TModel):
  """Attention baseline model for VQA."""

  @staticmethod
  def train_hooks():
    restore_resnet_hook = restore_hook.RestoreHook(
        # TODO(zichaoy): hard code the path given static function.
        checkpoint_path="/cns/lu-d/home/zichaoy/rs=6.3/resnet_v2_152.ckpt",
        new_model_scope="vqa_attention_baseline/body/",
        old_model_scope="resnet_v2_152/",
    )
    return [restore_resnet_hook]

  def body(self, features):
    hp = self.hparams
    image_feat = image_embedding(
        features["inputs"],
        trainable=hp.train_resnet,
        is_training=hp.mode == tf.estimator.ModeKeys.TRAIN)

    # apply layer normalization and dropout on image_feature
    image_feat = common_layers.layer_postprocess(None, image_feat, hp)
    # apply dropout on question embedding
    question = tf.nn.dropout(features["question"], 1. - hp.dropout)

    query = question_encoder(question, hp)
    image_ave = attn(image_feat, query, hp)

    image_question = tf.concat([image_ave, query], axis=1)
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

    max_question_length = hparams.max_question_length
    question = question[:, :max_question_length, :]
    actual_question_length = common_layers.shape_list(question)[1]
    length = tf.minimum(length, max_question_length)
    padding = [[0, 0],
               [0, max_question_length-actual_question_length],
               [0, 0]]
    question = tf.pad(question, padding)
    question_shape = question.get_shape().as_list()
    question_shape[1] = max_question_length
    question.set_shape(question_shape)

    question = [question[:, i, :] for i in range(max_question_length)]

    # rnn_layers = [_get_rnn_cell(hparams)
    #               for _ in range(hparams.num_rnn_layers)]
    # rnn_multi_cell = tf.contrib.rnn.MultiRNNCell(rnn_layers)
    rnn_cell = _get_rnn_cell(hparams)
    # outputs, _ = tf.nn.dynamic_rnn(
    #     rnn_cell, question, length, dtype=tf.float32)
    outputs, _ = tf.nn.static_rnn(rnn_cell, question, sequence_length=length,
                                  dtype=tf.float32)
    outputs = [tf.expand_dims(output, axis=1) for output in outputs]
    outputs = tf.concat(outputs, axis=1)

    batch_size = common_layers.shape_list(outputs)[0]
    row_indices = tf.range(batch_size)
    # length - 1 as index
    indices = tf.transpose([row_indices, tf.maximum(length-1, 0)])
    last_output = tf.gather_nd(outputs, indices)

  return last_output


def attn(image_feat, query, hparams, name="attn"):
  """Attention on image feature with question as query."""
  with tf.variable_scope(name, "attn", values=[image_feat, query]):
    attn_dim = hparams.attn_dim
    num_glimps = hparams.num_glimps
    num_channels = common_layers.shape_list(image_feat)[-1]
    image_feat = common_layers.flatten4d3d(image_feat)
    query = tf.expand_dims(query, 1)
    image_proj = common_attention.compute_attention_component(
        image_feat, attn_dim, name="image_proj")
    query_proj = common_attention.compute_attention_component(
        query, attn_dim, name="query_proj")
    h = tf.nn.relu(image_proj + query_proj)
    h_proj = common_attention.compute_attention_component(
        h, num_glimps, name="h_proj")
    p = tf.nn.softmax(h_proj, axis=1)
    image_ave = tf.matmul(image_feat, p, transpose_a=True)
    image_ave = tf.reshape(image_ave, [-1, num_channels*num_glimps])

    return image_ave


def mlp(feature, hparams, name="mlp"):
  """Multi layer perceptron with dropout and relu activation."""
  with tf.variable_scope(name, "mlp", values=[feature]):
    num_mlp_layers = hparams.num_mlp_layers
    mlp_dim = hparams.mlp_dim
    for i in range(num_mlp_layers):
      feature = common_layers.dense(feature, mlp_dim, activation=tf.nn.relu)
      feature = common_layers.dropout_with_broadcast_dims(
          feature, keep_prob=1-hparams.dropout, name="layer_%i"%(i))
    return feature


@registry.register_hparams
def vqa_attention_base():
  """VQA attention baseline hparams."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 2
  hparams.use_fixed_batch_size = True,
  hparams.optimizer = "Adam"
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.999
  hparams.optimizer_adam_epsilon = 1e-8
  hparams.weight_decay = 0
  hparams.clip_grad_norm = 0.
  hparams.initializer = "uniform_unit_scaling"
  hparams.initializer_gain = 2.
  hparams.learning_rate = 0.5
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_warmup_steps = 0
  hparams.learning_rate_decay_scheme = "exp"
  hparams.learning_rate_decay_rate = 0.5
  hparams.learning_rate_decay_steps = 50000

  # not used hparams
  hparams.label_smoothing = 0.
  hparams.multiply_embedding_mode = ""

  hparams.dropout = 0.5
  hparams.norm_type = "layer"
  hparams.layer_postprocess_sequence = "nd"
  hparams.layer_prepostprocess_dropout = 0.5

  # add new hparams
  # preprocess
  hparams.add_hparam("resize_side", 512)
  hparams.add_hparam("height", 448)
  hparams.add_hparam("width", 448)
  hparams.add_hparam("distort", True)

  hparams.add_hparam("train_resnet", False)
  hparams.add_hparam("rnn_type", "lstm")
  hparams.add_hparam("num_rnn_layers", 1)
  hparams.add_hparam("max_question_length", 15)
  # lstm hidden size
  hparams.hidden_size = 512

  hparams.add_hparam("attn_dim", 512)
  hparams.add_hparam("num_glimps", 2)

  hparams.add_hparam("num_mlp_layers", 1)
  hparams.add_hparam("mlp_dim", 1024)

  return hparams
