"""
Example command-line script invocation:
python i3d_estimator.py --model_dir C:/Users/Public/fra-gctd-project/Models/ramsey_nj/i3d/pretrained-init --pretrained_warm_start --monitor_steps 10 --learning_rate 0.1 --batch_size 2 --variables_to_train logits

COARSE_TUNE
python i3d_estimator.py --mode train --model_dir /media/data_0/fra/gctd/Models/ramsey_nj/i3d-estimator-pretrained-init --checkpoint_path /media/data_0/fra/gctd/Models/pre-trained/i3d_kinetics_600_rgb/model.ckpt --tfrecord_dir_path /media/data_0/fra/gctd/Data_Sets/ramsey_nj/seed_data_set_128 --num_gpus 2 --pretrained_warm_start --monitor_steps 50 --learning_rate 0.1 --batch_size 6 --variables_to_train logits

FINE_TUNE
python i3d_estimator.py --mode train --model_dir /media/data_0/fra/gctd/Models/ramsey_nj/i3d-estimator-pretrained-init-seed_data_set_128 --tfrecord_dir_path /media/data_0/fra/gctd/Data_Sets/ramsey_nj/seed_data_set_128 --num_gpus 2 --monitor_steps 50 --learning_rate 0.00001 --batch_size 6
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from metrics import f1 as f1_metric
from nets.i3d import *
import numpy as np
from os import cpu_count, path, putenv
from i3d_vars import i3d_vars

slim = tf.contrib.slim

# adapted from tf.slim train image classifier
def get_variables_to_train(trainable_scopes):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if trainable_scopes is None:
    return tf.trainable_variables()
  # The provided trainable scoped may be a list of strings or a single string
  # with comma-separated scopes
  scopes = [scope.strip() for scope in trainable_scopes.split(',')] \
    if isinstance(trainable_scopes, str) else trainable_scopes
  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

# The slim-based implementation of i3d expects its
# prediction_fn to accept a scope argument
def scoped_sigmoid(logits, scope=None):
  with tf.name_scope(scope):
    return tf.sigmoid(logits)

# The estimator API expects a single model_fn to support training, evaluation
# or prediction, depending on the mode passed to the model_fn by the estimator.
def i3d_fn(features, labels, mode, params):
  # Compute logits.
  with slim.arg_scope(i3d_arg_scope(weight_decay=params['weight_decay'])):
    logits, endpoints = i3d(
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

  # Compute predictions using round instead of argmax since our prediction
  # function is sigmoid (for multi-label classification) and not softmax
  # (for multi-class classification).
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

  # L1 loss is not included by default, but helps with our particular task
  for var in tf.trainable_variables():
    if var.op.name.find(r'weights') > 0 \
        and var not in tf.get_collection(tf.GraphKeys.WEIGHTS):
      tf.add_to_collection(tf.GraphKeys.WEIGHTS, var)

  l1_loss = tf.contrib.layers.apply_regularization(
    regularizer=tf.contrib.layers.l1_regularizer(scale=params['weight_decay']),
    weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
  tf.summary.scalar('Losses/l1_loss', l1_loss)

  # L2 loss is already computed when utilizing the slim argument scope,
  # including the weight decay arument. Just display the existing value
  l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  tf.summary.scalar('Losses/l2_loss', l2_loss)

  regularization_loss = tf.add(l1_loss, l2_loss)
  tf.summary.scalar('Losses/regularization_loss', regularization_loss)

  total_loss = tf.add(sigmoid_loss, regularization_loss)
  tf.summary.scalar('Losses/total_loss', total_loss)

  # Compute evaluation metrics.
  auc = tf.metrics.auc(
    labels=labels, predictions=predicted_classes, name='auc_op')

  precision = tf.metrics.precision(
    labels=labels, predictions=predicted_classes, name='precision_op')

  recall = tf.metrics.recall(
    labels=labels, predictions=predicted_classes, name='recall_op')

  f1 = f1_metric(labels=labels, predictions=predicted_classes, name='f1_op')

  if mode == tf.estimator.ModeKeys.EVAL:
    metrics = {
      'Metrics/eval/auc': auc,
      'Metrics/eval/f1': f1,
      'Metrics/eval/precision': precision,
      'Metrics/eval/recall': recall
    }

    return tf.estimator.EstimatorSpec(
      mode, loss=total_loss, eval_metric_ops=metrics)

  # Create training op.
  assert mode == tf.estimator.ModeKeys.TRAIN

  # Add summaries if we are training only and not evaluating
  # If evaluating, the estimator spec will add summaries automatically
  tf.summary.scalar('Metrics/train/auc', auc[1])
  tf.summary.scalar('Metrics/train/precision', precision[1])
  tf.summary.scalar('Metrics/train/recall', recall[1])
  tf.summary.scalar('Metrics/train/f1', f1[1])

  # Add histograms for variables.
  for variable in tf.global_variables():
    tf.summary.histogram(variable.op.name, variable)

  # prepare optimizer.
  if params['optimizer'] == 'momentum':
    #SGD + Momentum is the optimizer used to pre-train i3d
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=params['learning_rate'], momentum=params['momentum'])
  else:
    # pure SDG is a safe optimizer to use when troubleshooting problems
    # restoring Momentum variables from checkpoints using the Estimator API
    optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params['learning_rate'])

  variables_to_train = get_variables_to_train(params['variables_to_train'])

  train_op = tf.contrib.training.create_train_op(
    total_loss=total_loss, optimizer=optimizer,
    variables_to_train=variables_to_train)

  return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

# Parse the input tf.Example proto using the dictionary above.
feature_spec = {
  'feature': tf.FixedLenFeature([], tf.string),
  'label': tf.FixedLenFeature([], tf.string)
}

# Create a dictionary describing the features.
def parse_serialized_example(example_proto):
  tf.logging.info('example_proto: {}'.format(example_proto))
  # Parse the input tf.Example proto using the dictionary above.
  example = tf.parse_single_example(example_proto, feature_spec)
  tf.logging.info('example: {}'.format(example))
  return example['feature'], example['label']

def main(argv):
  args = parser.parse_args(argv[1:])

  # prepare to ingest the data set
  def preprocess_example(feature, label):
    feature = tf.decode_raw(feature, tf.uint8)
    feature = tf.reshape(feature, [args.clip_length, args.frame_height,
                                   args.frame_width, args.channels])
    feature = tf.image.convert_image_dtype(feature, dtype=tf.float32)
    feature = tf.subtract(feature, 0.5)
    feature = tf.multiply(feature, 2.0)
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [args.num_classes, ])
    label = tf.cast(label, tf.float32)
    return feature, label

  def get_train_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.train_subset_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset,
                                      buffer_size=args.tfrecord_size * 2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
      buffer_size=args.batch_size))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=preprocess_example, batch_size=args.batch_size,
      num_parallel_calls=cpu_count()))
    dataset = dataset.prefetch(args.prefetch_size)
    return dataset

  def get_eval_dataset():
    dataset = tf.data.Dataset.list_files(
      path.join(args.eval_subset_dir_path, '*.tfrecord'))
    dataset = tf.data.TFRecordDataset(dataset,
                                      buffer_size=args.tfrecord_size * 2 ** 20,
                                      num_parallel_reads=cpu_count())
    dataset = dataset.map(parse_serialized_example,
                          num_parallel_calls=cpu_count())
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
      map_func=preprocess_example, batch_size=args.batch_size,
      num_parallel_calls=cpu_count()))
    dataset = dataset.prefetch(args.prefetch_size)
    return dataset

  def get_test_dataset():
    pass



  # prepare to use zero or more GPUs
  if args.num_gpus == 1:
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.95)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
    distribute_strategy = None
    putenv('CUDA_VISIBLE_DEVICES', '{}'.format(args.gpu_num))
  elif args.num_gpus > 1:
    gpu_options = tf.GPUOptions(
      allow_growth=True, per_process_gpu_memory_fraction=.95)
    session_config = tf.ConfigProto(
      allow_soft_placement=True, gpu_options=gpu_options)
    # virtual gpu names are independent of device name
    devices = ['/gpu:{}'.format(i) for i in range(args.num_gpus)]
    # MirroredStrategy dependency NCCL not implemented on Windows
    distribute_strategy = tf.distribute.MirroredStrategy(devices=devices)

    # TODO: parameterize list of CUDA_VISIBLE_DEVICE numbers
    device_names = ''
    for i in range(args.num_gpus - 1):
      device_names += '{},'.format(i)
    device_names += '{}'.format(args.num_gpus - 1)
    putenv('CUDA_VISIBLE_DEVICES', device_names)
  else:  # just use the CPU
    session_config = None
    distribute_strategy = None
    putenv('CUDA_VISIBLE_DEVICES', '')

  # since training halts for validation to be performed, assign all available
  # resources to evaluation (e.g. use the same training distribution strategy)
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
  try:
    variables_to_warm_start = i3d_vars[args.variables_to_warm_start]
  except KeyError:
    variables_to_warm_start = '.*'

  initialization_ckpt = args.checkpoint_path if args.checkpoint_path \
    else args.model_dir

  warm_start_settings = tf.estimator.WarmStartSettings(
    ckpt_to_initialize_from=initialization_ckpt,
    vars_to_warm_start=variables_to_warm_start)

  try:
    variables_to_train = i3d_vars[args.variables_to_train]
  except KeyError:
    variables_to_train = None

  # create the model
  classifier = tf.estimator.Estimator(
    model_fn=i3d_fn,
    params={
      'num_classes': args.num_classes,
      'learning_rate': args.learning_rate,
      'optimizer': args.optimizer,
      'momentum': args.momentum,
      'dropout_rate': args.dropout_rate,
      'variables_to_train': variables_to_train,
      'weight_decay': args.weight_decay
    },
    config=estimator_config,
    warm_start_from=warm_start_settings)

  if args.mode == 'train_and_eval':
    # train and evaluate the model.
    tf.estimator.train_and_evaluate(
      estimator=classifier,
      train_spec=tf.estimator.TrainSpec(input_fn=get_train_dataset,
                                        max_steps=args.train_steps),
      eval_spec=tf.estimator.EvalSpec(input_fn=get_eval_dataset, steps=None,
                                      start_delay_secs=0, throttle_secs=0))
  elif args.mode == 'train':
    # train the model.
    classifier.train(input_fn=get_train_dataset, steps=args.train_steps)
  elif args.mode == 'eval':
    # evaluate the model.
    eval_result = classifier.evaluate(input_fn=get_eval_dataset)
    tf.logging.info(
      '\nTraining set metrics:\n\tauc: {auc:0.3f}\n\tprecision: '
      '{precision:0.3f}\n\trecall: {recall:0.3f}\n\tf1: {f1:0.3f}\n'.format(
        **eval_result))
  elif args.mode == 'predict':
    # Generate predictions from the model
    predictions = classifier.predict(input_fn=get_eval_dataset)

    labels = []

    with tf.Session().as_default() as sess:
      dataset = get_test_dataset()
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
  elif args.mode == 'export':
    def serving_input_receiver_fn():
      """An input receiver that expects a serialized tf.Example."""
      serialized_tf_example = tf.placeholder(dtype=tf.string, shape=[])

      parsed_features = tf.parse_single_example(
        serialized_tf_example, feature_spec)

      parsed_features['feature'], parsed_features['label'] = preprocess_example(
        parsed_features['feature'], parsed_features['label'])

      return tf.estimator.export.TensorServingInputReceiver(
        features=tf.expand_dims(parsed_features['feature'], 0),
        receiver_tensors=serialized_tf_example)

    classifier.export_saved_model(
      args.export_path, serving_input_receiver_fn=serving_input_receiver_fn,
      checkpoint_path=args.checkpoint_path)
  else:
    raise ValueError(
      '--mode parameter requires specification using an argument from the set'
        ' {\'train\', \'eval\', \'predict\'}')

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True,
                    help='train_and_eval, train, eval, predict or export')
parser.add_argument('--batch_size', default=6, type=int)
parser.add_argument('--prefetch_size', default=tf.data.experimental.AUTOTUNE,
                    type=int)
parser.add_argument('--monitor_steps', default=100, type=int)
parser.add_argument('--train_steps', default=None, type=int)
parser.add_argument('--num_classes', default=204, type=int)
parser.add_argument('--learning_rate', default=1e-1, type=float)
parser.add_argument('--optimizer', default='momentum', help='sgd or momentum')
parser.add_argument('--momentum', default=.9, type=float)
parser.add_argument('--weight_decay', default=1e-7, type=float)
parser.add_argument('--dropout_rate', default=.2, type=float)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--clip_length', default=64, type=int)
parser.add_argument('--frame_height', default=i3d.default_image_size, type=int)
parser.add_argument('--frame_width', default=i3d.default_image_size, type=int)
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--variables_to_warm_start', default=None)
parser.add_argument('--variables_to_train', default=None)
parser.add_argument('--train_subset_dir_path', default=None)
parser.add_argument('--eval_subset_dir_path', default=None)
parser.add_argument('--test_subset_dir_path', default=None)
parser.add_argument('--tfrecord_size', default=40, type=int,
                    help='approximate size of the given data set\'s tfrecords '
                         'in bytes')
parser.add_argument('--checkpoint_path',default=None)
parser.add_argument('--export_path',default=None)
parser.add_argument('--model_dir', required=True)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
