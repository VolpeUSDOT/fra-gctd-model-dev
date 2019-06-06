from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.metrics import precision, recall

def safe_divide(numerator, denominator):
  """Divides two values, returning 0 if the denominator is <= 0.
  Copied from the metric_ops.py protected member function.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return tf.where(
    tf.greater(denominator, 0), tf.truediv(numerator, denominator), 0)


@tf_export('metrics.f1')
def f1(labels,
       predictions,
       weights=None,
       metrics_collections=None,
       updates_collections=None,
       name=None):
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.precision is not '
                       'supported when eager execution is enabled.')

  with variable_scope.variable_scope(
      name, 'f1', (predictions, labels, weights)):
    prec, precision_update_op = precision(
      labels, predictions, weights=weights, metrics_collections=None,
      updates_collections=None, name=None)
    rec, recall_update_op = recall(
      labels, predictions, weights=weights, metrics_collections=None,
      updates_collections=None, name=None)

    def compute_f1(p, r, name):
      return tf.multiply(
        2., safe_divide(tf.multiply(p, r), tf.add(tf.multiply(1., p), r)), 
        name=name)

    value = compute_f1(prec, rec, 'value')
    update_op = compute_f1(precision_update_op, recall_update_op, 'update_op')

    if metrics_collections:
      ops.add_to_collections(metrics_collections, value)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return value, update_op
