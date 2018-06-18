"""Data generators for VQA data sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import zipfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry

import tensorflow as tf

_MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
_MSCOCO_IMAGE_URLS = [
    "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
]
_VQA_V2_ANNOTATION_URL = "xxx"  # TODO(zichaoy) to be filled


def _get_vqa_v2_dataset(directory):
  """Extract the VQA V2 data set to directory unless it's there."""
  for url in _MSCOCO_IMAGE_URLS:
    filename = os.path.basename(url)
    download_url = os.path.join(_MSCOCO_ROOT_URL, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)

  annotation_file = generator_utils.maybe_download_drom_drive(
      directory, "vqa_v2.tar.gz", _VQA_V2_ANNOTATION_URL)
  with tarfile.open(annotation_file, "r:gz") as annotation_tar:
    annotation_tar.extractall(directory)


class ImageQuestion2MultilabelProblem(image_utils.ImageProblem):
  """Base class for image question answer problem."""

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def question_vocab_size(self):
    raise NotImplementedError

  @property
  def num_classes(self):
    raise NotImplementedError()

  @property
  def vocab_filename(self):
    raise NotImplementedError()

  @property
  def label_filename(self):
    raise NotImplementedError()

  @property
  def train_shards(self):
    raise NotImplementedError()

  @property
  def dev_shards(self):
    raise NotImplementedError()

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (
      super(image_utils.ImageProblem, self).example_reading_spec())
    data_fields["image/image_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question"] = tf.FixedLenSequenceFeature((), tf.int64)
    data_fields["image/answer"] = tf.FixedLenSequenceFeature((), tf.int64)
    # TODO(zichaoy): maybe add answer len?
    # depends on the implementation of multiclass modality

    data_items_to_decoders["question"] \
      = tf.contrib.slim.tf.example_decoder.Tensor("image/question")
    data_items_to_decoders["targets"] \
      = tf.contrib.slim.tfexample_decoder.Tensor("image/answer")
    return data_fields, data_items_to_decoders

  def feature_encoders(self, data_dir):
    input_encoder = text_encoder.ImageEncoder(channels=self.num_channels)
    vocab_file = os.path.join(data_dir, self.vocab_filename)
    question_encoder = text_encoder.TextTokenEncoder(vocab_file)
    label_file = os.path.join(data_dir, self.label_filename)
    target_encoder = text_encoder.ClassLabelEncoder(
      class_labels_fname=label_file)
    return {"inputs": input_encoder,
            "question": question_encoder,
            "targets": target_encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    encoder = self._encoders["question"]
    p.input_modality = {"inputs": (registry.Modalities.IMAGE, 256),
                        "question" (registry.Modalities.SYMBOL, encoder.vocab_size)}
    # TODO(zichaoy): set batch_size multiplier, loss multiplier ?
    p.input_space_id = problem.SpaceID.IMAGE  # multiple input features?
    p.target_space_id = self.target_space_id

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
      self.generator(data_dir, tmp_dir, True),
      self.training_filepaths(data_dir, self.train_shards, shuffled=False),
      self.generator(data_dir, tmp_dir, False),
      self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))

@registry.register_problem
class ImageVqav2Tokens10kLabels3k(ImageQuestion2MultilabelProblem):
  """VQA V2, 10k question vocab, 3k answer label."""

  @property
  def question_vocab_size(self):
    return 10000

  @property
  def answer_num_classes(self):
    return 3000

  @property
  def question_vocab_fileame(self):
    return "question.vocab.%d" % self.question_vocab_size

  @property
  def answer_label_filename(self):
    return "answer.label.%d" % self.answer_num_classes


