from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from metrics import f_beta_measure
from nets.s3dg import *
import numpy as np
from os import cpu_count, path, putenv
from vars_to_warm_start import vars_to_warm_start
from vars_to_train import vars_to_train


slim = tf.contrib.slim

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--monitor_steps', default=1, type=int)
parser.add_argument('--train_steps', default=100000, type=int)
parser.add_argument('--num_classes', default=204, type=int)
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--optimizer', default='sgd', help='sgd or momentum')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--num_gpus', default=2, type=int)
parser.add_argument('--length', default=64, type=int)
parser.add_argument('--height', default=s3dg.default_image_size, type=int)
parser.add_argument('--width', default=s3dg.default_image_size, type=int)
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--pretrained_warm_start', action='store_true')
parser.add_argument('--variables_to_train', default=None)
parser.add_argument('--variables_to_warm_start', default='mixed_5')
parser.add_argument('--variables_to_exclude', default='mixed_5')
parser.add_argument('--tfrecord_dir_path',
                    default='/media/data_0/fra/gctd/Data_Sets/ramsey_nj_2x')
parser.add_argument('--checkpoint_path',
                    default='/media/data_0/fra/gctd/Models/pre-trained/'
                            's3dg_kinetics_600_rgb/model.ckpt')
parser.add_argument('--model_dir',
                    default='/media/data_0/fra/gctd/Models/ramsey_nj/'
                            's3dg-pretrained-init')

def get_variables_to_train(trainable_scopes):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

# The slim-based implementation of s3dg expects its
# prediction_fn to accept a scope argument
def scoped_sigmoid(logits, scope=None):
  with tf.name_scope(scope):
    return tf.sigmoid(logits)

# The estimator API expects a single model_fn to support training, evaluation,
# and prediction, depending on the mode passed to model_fn by the estimator.
def s3dg_fn(features, labels, mode, params):
  # Compute logits.
  with slim.arg_scope(s3dg_arg_scope()):
    logits, endpoints = s3dg(
      features,
      num_classes=params['num_classes'],
      dropout_keep_prob=1. - params['dropout_rate'],
      is_training=True if mode == tf.estimator.ModeKeys.TRAIN else False,
      prediction_fn=scoped_sigmoid)

  # Add summaries for end_points.
  for endpoint in endpoints:
    x = endpoints[endpoint]
    tf.summary.histogram('activations/' + endpoint, x)
    tf.summary.scalar('sparsity/' + endpoint, tf.nn.zero_fraction(x))

  # Compute predictions.
  predicted_classes = tf.round(endpoints['Predictions'])

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
      'class_ids': predicted_classes,
      'probabilities': endpoints['Predictions'],
      'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Compute primary loss.
  sigmoid_loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  tf.summary.scalar('Losses/sigmoid_loss', sigmoid_loss)

  # Regularization loss is already computed when utilizing the slim argument
  # scope, including the weight decay arument. Just display the existing value
  regularization_loss = tf.reduce_sum(tf.get_collection(
    tf.GraphKeys.REGULARIZATION_LOSSES))
  tf.summary.scalar('Losses/regularization_loss', regularization_loss)

  total_loss = tf.add(sigmoid_loss, regularization_loss)
  tf.summary.scalar('Losses/total_loss', total_loss)

  # Compute evaluation metrics.
  auc = tf.metrics.auc(
    labels=labels, predictions=predicted_classes, name='auc_op')
  tf.summary.scalar('Metrics/auc', auc[1])

  precision = tf.metrics.precision(
    labels=labels, predictions=predicted_classes, name='precision_op')
  tf.summary.scalar('Metrics/precision', precision[1])

  recall = tf.metrics.recall(
    labels=labels, predictions=predicted_classes, name='recall_op')
  tf.summary.scalar('Metrics/recall', recall[1])

  f1 = f_beta_measure(precision[1], recall[1])
  tf.summary.scalar('Metrics/f1', f1)

  # Add histograms for variables.
  for variable in tf.global_variables():
    tf.summary.histogram(variable.op.name, variable)

  if mode == tf.estimator.ModeKeys.EVAL:
    metrics = {
      'auc': auc,
      'precision': precision,
      'recall': recall,
      'f1': f1
    }

    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  # prepare optimizer
  if params['optimizer'] == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=params['learning_rate'], momentum=params['momentum'])
  else:
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])

  variables_to_train = get_variables_to_train(params['variables_to_train'])
  train_op = optimizer.minimize(total_loss,
                                global_step=tf.train.get_global_step(),
                                var_list=variables_to_train)

  return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

# Create a dictionary describing the features.
image_feature_description = {
  'feature': tf.FixedLenFeature([], tf.string),
  'label': tf.FixedLenFeature([], tf.string)
}

def parse_serialized_example(example_proto):
  tf.logging.info('example_proto: {}'.format(example_proto))
  # Parse the input tf.Example proto using the dictionary above.
  example = tf.parse_single_example(example_proto, image_feature_description)
  tf.logging.info('example: {}'.format(example))
  return example['feature'], example['label']

def main(argv):
  args = parser.parse_args(argv[1:])

  # prepare to ingest the data set
  def preprocess_example(feature, label):
    feature = tf.decode_raw(feature, tf.uint8)
    feature = tf.reshape(
      feature, [args.length, args.height, args.width, args.channels])
    feature = tf.image.convert_image_dtype(feature, dtype=tf.float32)
    feature = tf.subtract(feature, 0.5)
    feature = tf.multiply(feature, 2.0)

    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [args.num_classes, ])
    label = tf.cast(label, tf.float32)

    return feature, label

  def get_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.tfrecord_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset, buffer_size=2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.map(preprocess_example,
                          num_parallel_calls=cpu_count())
    return dataset

  def get_train_dataset():
    dataset = get_dataset()
    dataset = dataset.shuffle(buffer_size=150)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=args.batch_size)
    return dataset

  def get_eval_dataset():
    dataset = get_dataset()
    dataset = dataset.batch(batch_size=args.batch_size)
    return dataset

  # prepare to use zero or more GPUs
  if args.num_gpus == 1:
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.95)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
    devices = ['/gpu:0']  # virtual gpu names are independent of device names
    distribute_strategy = tf.contrib.distribute.MirroredStrategy(
      devices=devices)
    putenv('CUDA_VISIBLE_DEVICES', '{}'.format(args.gpu_num))
  elif args.num_gpus > 1:  # TODO: parameterize list of CUDA_VISIBLE_DEVICE nums
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.95)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
    devices = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]
    distribute_strategy = tf.contrib.distribute.MirroredStrategy(
      devices=devices)
    device_names = ''
    for i in range(args.num_gpus - 1):
      device_names += '{},'.format(i)
    device_names += '{},'.format(args.num_gpus - 1)
    putenv('CUDA_VISIBLE_DEVICES', device_names)
  else:  # just use the CPU
    session_config = None
    distribute_strategy = None
    putenv('CUDA_VISIBLE_DEVICES', '')


  estimator_config = tf.estimator.RunConfig(
    model_dir=args.model_dir,
    save_summary_steps=args.monitor_steps,
    save_checkpoints_steps=args.monitor_steps,
    session_config=session_config,
    keep_checkpoint_max=10000,
    keep_checkpoint_every_n_hours=10000,
    log_step_count_steps=args.monitor_steps,
    train_distribute=distribute_strategy,
    eval_distribute=distribute_strategy)

  # prepare to restore weights from an existing checkpoint
  warm_start_vars = vars_to_warm_start[args.variables_to_warm_start] \
    if (args.warm_start or args.pretrained_warm_start) else '.*'

  warm_start_settings = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from=args.checkpoint_path,
    vars_to_warm_start=warm_start_vars) \
    if (args.warm_start or args.pretrained_warm_start) else None

  try:
    variables_to_train = vars_to_train[args.variables_to_train]
  except KeyError:
    variables_to_train = None

  # create the model
  classifier = tf.estimator.Estimator(
    model_fn=s3dg_fn,
    params={
      'num_classes': args.num_classes,
      'learning_rate': args.learning_rate,
      'optimizer': args.optimizer,
      'momentum': args.momentum,
      'dropout_rate': args.dropout_rate,
      'variables_to_train': variables_to_train
    },
    config=estimator_config,
    warm_start_from=warm_start_settings)

  # train the model.
  classifier.train(input_fn=get_train_dataset, steps=args.train_steps)

  # evaluate the model.
  eval_result = classifier.evaluate(input_fn=get_eval_dataset)
  tf.logging.info(
    '\nTraining set metrics:\n\tauc: {auc:0.3f}\n\tprecision: '
    '{precision:0.3f}\n\trecall: {recall:0.3f}\n\tf1: {f1:0.3f}\n'.format(
      **eval_result))

  # Generate predictions from the model
  predictions = classifier.predict(input_fn=get_eval_dataset)

  labels = []

  with tf.Session().as_default() as sess:
    dataset = get_dataset()
    iterator = dataset.make_one_shot_iterator()
    get_next = iterator.get_next()

    while True:
      try:
        labels.append(sess.run(get_next)[1])
      except tf.errors.OutOfRangeError:
        break

  for pred_dict, expec in zip(predictions, labels):
    probability = pred_dict['probabilities']
    tf.logging.info('\nPrediction is\n\t{},\nexpected\n\t{}'.format(
      probability, expec))
    tf.logging.info('\nDifference is\n\t{}'.format(np.not_equal(
      probability, expec)))
    tf.logging.info('\nDifference is\n\t{}'.format(np.arange(args.num_classes)))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
