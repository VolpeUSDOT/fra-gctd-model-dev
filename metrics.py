from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.metrics import precision, recall
from tensorflow.python.ops.metrics_impl import _aggregate_across_replicas
from tensorflow.python.ops.metrics_impl import _remove_squeezable_dimensions
from tensorflow.python.ops.metrics_impl import _safe_scalar_div


# inspired by tensorflow.python.ops.metrics_impl.precision
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
    predictions, labels, weights = _remove_squeezable_dimensions(
      predictions=math_ops.cast(predictions, dtype=dtypes.bool),
      labels=math_ops.cast(labels, dtype=dtypes.bool), weights=weights)

    prec, precision_update_op = precision(
      labels, predictions, weights, metrics_collections=None,
      updates_collections=None, name=None)

    rec, recall_update_op = recall(
      labels, predictions, weights, metrics_collections=None,
      updates_collections=None, name=None)

    def compute_f1(p, r, name):
      return math_ops.multiply(2., _safe_scalar_div(math_ops.multiply(
        p, r), math_ops.add(math_ops.multiply(1., p), r), 'f1'), name)

    def once_across_replicas(_, true_p, false_p):
      return compute_f1(true_p, false_p, 'value')

    value = _aggregate_across_replicas(
      metrics_collections, once_across_replicas, prec, rec)

    update_op = compute_f1(precision_update_op, recall_update_op, 'update_op')

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return value, update_op
