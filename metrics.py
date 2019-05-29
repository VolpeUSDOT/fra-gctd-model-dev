import tensorflow as tf

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


def f_beta_measure(precision, recall, beta=1.0):
  beta_squared = tf.multiply(beta, beta)
  f_value = tf.multiply(
    tf.add(1.0, beta_squared),
    safe_divide(
      tf.multiply(precision, recall),
      tf.add(
        tf.multiply(beta_squared, precision), recall
      )
    )
  )
  return f_value
