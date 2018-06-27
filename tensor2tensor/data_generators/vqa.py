"""Data generators for VQA data sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import tarfile
import zipfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.google.data_generators import vqa_utils
from tensor2tensor.utils import registry

import tensorflow as tf

_MSCOCO_ROOT_URL = "http://msvocds.blob.core.windows.net/"
_MSCOCO_IMAGE_URLS = [
    "coco2014/train2014.zip", "coco2014/val2014.zip", "coco2014/test2014.zip",
]
_VQA_V2_ANNOTATION_URL = ("https://drive.google.com/uc?export=download&id="
                          "1xfMU54ObCLvMRAekT3cfcIg-AgY39fWB")


def _get_vqa_v2_dataset(directory):
  """Extract the VQA V2 data set to directory unless it's there."""
  for url in _MSCOCO_IMAGE_URLS:
    filename = os.path.basename(url)
    download_url = os.path.join(_MSCOCO_ROOT_URL, url)
    path = generator_utils.maybe_download(directory, filename, download_url)
    unzip_dir = os.path.join(directory, filename.strip(".zip"))
    if not tf.gfile.Exists(unzip_dir):
      zipfile.ZipFile(path, "r").extractall(directory)

  annotation_file = generator_utils.maybe_download_from_drive(
      directory, "vqa_v2.tar.gz", _VQA_V2_ANNOTATION_URL)
  with tarfile.open(annotation_file, "r:gz") as annotation_tar:
    annotation_tar.extractall(directory)


def vqa_v2_generator(data_dir,
                     tmp_dir,
                     datasets,
                     vocab_filename,
                     label_filename,
                     eos_list=None):
  """vqa v2 generator."""
  eos_list = eos_list if eos_list else []
  _get_vqa_v2_dataset(tmp_dir)
  vocab_path = os.path.join(data_dir, vocab_filename)
  if not tf.gfile.Exists(vocab_path):
    vocab_tmp_path = os.path.join(tmp_dir, vocab_filename)
    tf.gfile.Copy(vocab_tmp_path, vocab_path)
    with tf.gfile.GFile(vocab_path, mode="r") as f:
      vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
    with tf.gfile.GFile(vocab_path, mode="w") as f:
      f.write(vocab_data)
  label_path = os.path.join(data_dir, label_filename)
  if not tf.gfile.Exists(label_path):
    label_tmp_path = os.path.join(tmp_dir, label_filename)
    tf.gfile.Copy(label_tmp_path, label_path)

  vocab_encoder = text_encoder.TokenTextEncoder(vocab_path, replace_oov="UNK")
  label_encoder = text_encoder.ClassLabelEncoder(class_labels_fname=label_path)

  prefix_annotation = []
  for prefix, annotation_file in datasets:
    annotation_path = os.path.join(tmp_dir, annotation_file)
    with tf.gfile.Open(annotation_path) as f:
      annotation_json = json.loads(f.read())
    prefix_annotation += [(prefix, anno) for anno in annotation_json]
  random.shuffle(prefix_annotation)
  annotation_count = len(prefix_annotation)
  tf.logging.info("Processing %d annotations for vqa v2" %(annotation_count))

  for prefix, anno in prefix_annotation:
    image_id = anno["image_id"]
    question = vocab_encoder.encode(anno["question"]) + eos_list
    answer = [label_encoder.encode(ans) for ans in anno["answer"]]
    answer = answer if answer else [0]  # 0 indicates padding
    image_filename = "COCO_" + prefix + "_" + str(image_id).zfill(12) + ".jpg"
    image_filepath = os.path.join(tmp_dir, prefix, image_filename)
    with tf.gfile.Open(image_filepath, "r") as f:
      encoded_image_data = f.read()
      yield {
          "image/encoded": [encoded_image_data],
          "image/format": ["jpeg"],
          "image/image_id": [image_id],
          "image/question_id": [anno["question_id"]],
          "image/question": question,
          "image/answer": answer,
      }


class ImageQuestion2MultilabelProblem(image_utils.ImageProblem):
  """Base class for image question answer problem."""

  @property
  def source_data_files(self, dataset_split):
    raise NotImplementedError()

  @property
  def target_space_id(self):
    raise NotImplementedError()

  @property
  def vocab_size(self):
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

  def generator(self, data_dir, tmp_dir, dataset_split):
    raise NotImplementedError()

  def example_reading_spec(self):
    data_fields, data_items_to_decoders = (
        super(image_utils.ImageProblem, self).example_reading_spec())
    data_fields["image/image_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question_id"] = tf.FixedLenFeature((), tf.int64)
    data_fields["image/question"] = tf.FixedLenSequenceFeature((), tf.int64)
    data_fields["image/answer"] = tf.FixedLenSequenceFeature((), tf.int64)

    data_items_to_decoders[
        "question"] = tf.contrib.slim.tf.example_decoder.Tensor(
            "image/question")
    data_items_to_decoders[
        "targets"] = tf.contrib.slim.tfexample_decoder.Tensor(
            "image/answer")
    return data_fields, data_items_to_decoders

  def feature_encoders(self, data_dir):
    input_encoder = text_encoder.ImageEncoder(channels=self.num_channels)
    vocab_file = os.path.join(data_dir, self.vocab_filename)
    question_encoder = text_encoder.TokenTextEncoder(
        vocab_file, replace_oov="UNK")
    label_file = os.path.join(data_dir, self.label_filename)
    target_encoder = text_encoder.ClassLabelEncoder(
        class_labels_fname=label_file)
    return {"inputs": input_encoder,
            "question": question_encoder,
            "targets": target_encoder}

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    encoder = self._encoders["question"]
    p.input_modality = {"inputs":
                        (registry.Modalities.IMAGE, 256),
                        "question":
                        (registry.Modalities.SYMBOL, encoder.vocab_size)}
    p.target_modality = (registry.Modalities.CLASS_LABEL + ":multi_label",
                         self.num_classes)
    # TODO(zichaoy): set batch_size multiplier, loss multiplier ?
    p.input_space_id = problem.SpaceID.IMAGE  # multiple input features?
    p.target_space_id = self.target_space_id

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    generator_utils.generate_dataset_and_shuffle(
        self.generator(data_dir, tmp_dir, problem.DatasetSplit.TRAIN),
        self.training_filepaths(data_dir, self.train_shards, shuffled=False),
        self.generator(data_dir, tmp_dir, problem.DatasetSplit.EVAL),
        self.dev_filepaths(data_dir, self.dev_shards, shuffled=False))


@registry.register_problem
class ImageVqav2Tokens10kLabels3k(ImageQuestion2MultilabelProblem):
  """VQA V2, 10k question vocab, 3k answer label."""
  _VQA_V2_TRAIN_DATASETS = [
      ("train2014", "v2_train2014_annotations.json"),
  ]
  _VQA_V2_DEV_DATASETS = [
      ("val2014", "v2_val2014_annotations.json"),
  ]
  _VQA_V2_TEST_DATASETS = [
      ("test2015", "v2_test2015_annotations.json"),
  ]

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return self._VQA_V2_TRAIN_DATASETS if train else self._VQA_V2_DEV_DATASETS

  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def vocab_size(self):
    return 10000

  @property
  def num_classes(self):
    return 3000

  @property
  def vocab_filename(self):
    return "question.vocab.%d" % self.vocab_size

  @property
  def label_filename(self):
    return "answer.label.%d" % self.num_classes

  @property
  def train_shards(self):
    return 128

  @property
  def dev_shards(self):
    return 64

  def preprocess_example(self, example, mode, hparams):
    # TODO(zichaoy) hparams? problem_hparams or model hparmas??
    image = example["inputs"]
    return vqa_utils.vqa_v2_preprocess_image(image, hparams.height,
                                             hparams.width, mode)

  def generator(self, data_dir, tmp_dir, dataset_split):
    datasets = self.source_data_files(dataset_split)
    return vqa_v2_generator(data_dir, tmp_dir, datasets,
                            self.vocab_filename, self.label_filename)
