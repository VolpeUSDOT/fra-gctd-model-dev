"""Stdlib-only converter security regression tests.

From PhaseOne, run:
  python -B -m unittest datasets.dataset_conversion_security_test -v
"""

from contextlib import ExitStack
import importlib
import os
import sys
import types
import unittest
from unittest import mock


class DatasetConversionSecurityTest(unittest.TestCase):

  def test_rejected_archive_stops_conversion_and_passes_trust_parameters(self):
    tensorflow = mock.MagicMock()
    numpy = mock.Mock()
    six = types.ModuleType('six')
    six.__path__ = []
    moves = types.ModuleType('six.moves')
    moves.cPickle = mock.Mock()
    six.moves = moves
    datasets = types.ModuleType('datasets')
    datasets.__path__ = [os.path.dirname(__file__)]
    cases = (
        ('download_and_convert_flowers', 'flower_photos',
         ('_get_filenames_and_classes', '_convert_dataset', 'ImageReader')),
        ('download_and_convert_cifar10', 'cifar-10-batches-py',
         ('_add_to_tfrecord',)),
    )

    # Replace the package too, so scoped imports cannot leave mocked dependencies
    # attached to the real datasets package when other tests run in this process.
    with mock.patch.dict(sys.modules, {
        'datasets': datasets,
        'tensorflow': tensorflow,
        'numpy': numpy,
        'six': six,
        'six.moves': moves,
    }):
      for name in ('dataset_utils', 'download_and_convert_flowers',
                   'download_and_convert_cifar10'):
        sys.modules.pop('datasets.' + name, None)

      for name, archive_root, conversion_steps in cases:
        with self.subTest(converter=name), ExitStack() as stack:
          converter = importlib.import_module('datasets.' + name)
          tensorflow.reset_mock()
          numpy.reset_mock()
          moves.cPickle.reset_mock()
          dataset_dir = 'unused-dataset'
          tensorflow.gfile.Exists.side_effect = (
              lambda filename: filename == dataset_dir)
          rejection = ValueError('archive rejected before conversion')
          downloader = stack.enter_context(mock.patch.object(
              converter.dataset_utils, 'download_and_uncompress_tarball',
              side_effect=rejection))
          blocked = [stack.enter_context(mock.patch.object(converter, step))
                     for step in conversion_steps +
                     ('_clean_up_temporary_files',)]
          image_to_tfexample = stack.enter_context(mock.patch.object(
              converter.dataset_utils, 'image_to_tfexample'))
          write_label_file = stack.enter_context(mock.patch.object(
              converter.dataset_utils, 'write_label_file'))

          with self.assertRaises(ValueError) as raised:
            converter.run(dataset_dir)

          self.assertIs(rejection, raised.exception)
          downloader.assert_called_once_with(
              converter._DATA_URL, dataset_dir, converter._DATA_SHA256,
              archive_root)
          for operation in blocked:
            operation.assert_not_called()
          tensorflow.python_io.TFRecordWriter.assert_not_called()
          tensorflow.gfile.Open.assert_not_called()
          tensorflow.gfile.FastGFile.assert_not_called()
          tensorflow.gfile.MakeDirs.assert_not_called()
          tensorflow.gfile.DeleteRecursively.assert_not_called()
          tensorflow.Graph.assert_not_called()
          tensorflow.Session.assert_not_called()
          tensorflow.image.decode_jpeg.assert_not_called()
          tensorflow.image.encode_png.assert_not_called()
          moves.cPickle.load.assert_not_called()
          numpy.squeeze.assert_not_called()
          image_to_tfexample.assert_not_called()
          write_label_file.assert_not_called()


if __name__ == '__main__':
  unittest.main()
