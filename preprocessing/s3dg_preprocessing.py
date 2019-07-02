import tensorflow as tf
from preprocessing.inception_preprocessing import apply_with_random_selector


# rather than modify inception_preprocessing, we copy the snippets we need here
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
      else:
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_brightness(image, max_delta=8. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_video(
    video, label, num_classes, length, height, width, channels,
    is_training=False, fast_mode=False):
  video = tf.decode_raw(video, tf.uint8)
  video = tf.reshape(video, [length, height, width, channels])
  video = tf.image.convert_image_dtype(video, dtype=tf.float32)

  if is_training:
    num_distort_cases = 1 if fast_mode else 4
    video = apply_with_random_selector(
      video,
      lambda x, ordering: distort_color(x, ordering, fast_mode),
      num_cases=num_distort_cases)

  video = tf.subtract(video, 0.5)
  video = tf.multiply(video, 2.0)

  # leave label dtype equal to uint8 to avoid underflow when evaluating
  label = tf.decode_raw(label, tf.uint8)
  label = tf.reshape(label, [num_classes, ])

  return video, label