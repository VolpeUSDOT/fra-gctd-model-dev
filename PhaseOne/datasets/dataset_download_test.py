"""Security regression tests requiring only the Python standard library."""

import hashlib
import io
import os
from pathlib import Path
import ssl
import tarfile
import tempfile
import unittest
from unittest import mock
from urllib import request

from datasets import dataset_download


class DatasetDownloadTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.dataset = Path(self.temporary.name) / 'dataset'
    self.dataset.mkdir()

  def archive(self, members):
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode='w:gz') as archive:
      for member, data in members:
        archive.addfile(member, io.BytesIO(data))
    return contents.getvalue()

  def file(self, name, data=b'marker'):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    return member, data

  def download(self, contents, digest=None, url='https://example.test/data.tgz',
               root='flower_photos', final_url=None):
    response = io.BytesIO(contents)
    response.geturl = lambda: final_url or url
    opener = mock.Mock()
    opener.open.return_value = response
    with mock.patch.object(dataset_download.request, 'build_opener',
                           return_value=opener):
      dataset_download.download_and_uncompress_tarball(
          url, str(self.dataset), digest or hashlib.sha256(contents).hexdigest(),
          root)

  def assertRejected(self, contents, **kwargs):
    before = sorted(str(path.relative_to(self.dataset))
                    for path in self.dataset.rglob('*'))
    with self.assertRaises((ValueError, tarfile.TarError, EOFError, OSError)):
      self.download(contents, **kwargs)
    after = sorted(str(path.relative_to(self.dataset))
                   for path in self.dataset.rglob('*'))
    self.assertEqual(before, after)

  def test_benign_nested_files_and_cleanup(self):
    directory = tarfile.TarInfo('flower_photos/daisy/')
    directory.type = tarfile.DIRTYPE
    contents = self.archive([
        (directory, b''), self.file('flower_photos/daisy/image.jpg')])
    self.download(contents)
    self.assertEqual(b'marker',
                     (self.dataset / 'flower_photos/daisy/image.jpg').read_bytes())
    self.assertEqual(['flower_photos'], os.listdir(str(self.dataset)))

  def test_cifar_root_and_cleanup(self):
    self.download(self.archive([self.file('cifar-10-batches-py/data_batch_1')]),
                  root='cifar-10-batches-py')
    self.assertEqual(b'marker',
                     (self.dataset / 'cifar-10-batches-py/data_batch_1').read_bytes())

  def test_download_uses_certificate_verification_and_redirect_policy(self):
    contents = self.archive([self.file('flower_photos/file')])
    response = io.BytesIO(contents)
    response.geturl = lambda: 'https://example.test/data.tgz'
    opener = mock.Mock()
    opener.open.return_value = response
    with mock.patch.object(dataset_download.request, 'build_opener',
                           return_value=opener) as build:
      dataset_download.download_and_uncompress_tarball(
          'https://example.test/data.tgz', str(self.dataset),
          hashlib.sha256(contents).hexdigest(), 'flower_photos')
    https, redirect = build.call_args[0]
    self.assertIsInstance(https, request.HTTPSHandler)
    self.assertEqual(ssl.CERT_REQUIRED, https._context.verify_mode)
    self.assertTrue(https._context.check_hostname)
    self.assertIsInstance(redirect, dataset_download._HTTPSOnlyRedirectHandler)

  def test_archive_permissions_are_not_applied(self):
    directory = tarfile.TarInfo('flower_photos')
    directory.type = tarfile.DIRTYPE
    directory.mode = 0
    member, data = self.file('flower_photos/file')
    member.mode = 0o4777
    self.download(self.archive([(directory, b''), (member, data)]))
    target = self.dataset / 'flower_photos/file'
    self.assertEqual(b'marker', target.read_bytes())
    if os.name != 'nt':
      self.assertEqual(0, target.stat().st_mode & 0o7111)

  def test_rejects_insecure_or_credentialed_urls_before_download(self):
    for url in ('http://example.test/data.tgz', 'file:///tmp/data.tgz',
                'ftp://example.test/data.tgz',
                'https://user:password@example.test/data.tgz'):
      with self.subTest(url=url), mock.patch.object(
          dataset_download.request, 'build_opener') as opener:
        self.assertRejected(self.archive([self.file('flower_photos/file')]),
                            url=url)
        opener.assert_not_called()

  def test_rejects_insecure_response_and_cleans_download(self):
    self.assertRejected(self.archive([self.file('flower_photos/file')]),
                        final_url='http://example.test/data.tgz')

  def test_redirect_handler_rejects_downgrade_before_request(self):
    handler = dataset_download._HTTPSOnlyRedirectHandler()
    original = request.Request('https://example.test/data.tgz')
    for url in ('http://example.test/data.tgz', 'file:///tmp/data.tgz',
                'https://user:password@example.test/data.tgz'):
      with self.subTest(url=url), self.assertRaises(ValueError):
        handler.redirect_request(original, None, 302, 'Found', {}, url)
    redirected = handler.redirect_request(
        original, None, 302, 'Found', {}, 'https://mirror.test/data.tgz')
    self.assertEqual('https://mirror.test/data.tgz', redirected.full_url)

  def test_wrong_digest_rejected_before_archive_open(self):
    with mock.patch.object(dataset_download.tarfile, 'open') as archive:
      self.assertRejected(b'not an archive', digest='0' * 64)
      archive.assert_not_called()

  def test_invalid_digest_rejected_before_download(self):
    for digest in ('missing', 'A' * 64, '0' * 63):
      with self.subTest(digest=digest), mock.patch.object(
          dataset_download.request, 'build_opener') as opener:
        self.assertRejected(b'archive', digest=digest)
        opener.assert_not_called()

  def test_truncated_archive_with_expected_digest_rejected(self):
    contents = self.archive([self.file('flower_photos/file')])
    self.assertRejected(contents[:len(contents) // 2],
                        digest=hashlib.sha256(contents).hexdigest())

  def test_invalid_archive_with_matching_digest_is_cleaned(self):
    self.assertRejected(b'not a gzip archive')

  def test_paths_are_rejected_and_prior_staged_files_removed(self):
    outside = self.dataset.parent / 'outside.txt'
    outside.write_bytes(b'unchanged')
    names = ('../outside.txt', str(outside), 'C:/outside.txt',
             '\\\\server\\share\\outside.txt',
             'flower_photos/../../outside.txt',
             'flower_photos/../flower_photos_evil/file',
             'flower_photos/./file', 'flower_photos//file',
             'flower_photos/dir\\outside.txt', 'flower_photos/C:/outside.txt',
             'flower_photos/file:stream', 'flower_photos/CON',
             'flower_photos/file.', 'flower_photos/file ',
             'flower_photos/file\nname',
             'different_root/file')
    for name in names:
      with self.subTest(name=name):
        self.assertRejected(self.archive([
            self.file('flower_photos/good'), self.file(name)]))
        self.assertEqual(b'unchanged', outside.read_bytes())

  def test_links_and_special_files_are_rejected(self):
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                 tarfile.CHRTYPE, tarfile.BLKTYPE):
      with self.subTest(kind=kind):
        member = tarfile.TarInfo('flower_photos/escape')
        member.type = kind
        member.linkname = '../../outside.txt'
        self.assertRejected(self.archive([
            (member, b''), self.file('flower_photos/escape/file')]))

  def test_duplicate_members_are_rejected(self):
    self.assertRejected(self.archive([
        self.file('flower_photos/file'), self.file('flower_photos/file')]))

  def test_existing_dataset_root_refused_before_download(self):
    root = self.dataset / 'flower_photos'
    root.mkdir()
    marker = root / 'unchanged'
    marker.write_bytes(b'original')
    with mock.patch.object(dataset_download.request, 'build_opener') as opener:
      self.assertRejected(self.archive([self.file('flower_photos/file')]))
      opener.assert_not_called()
    self.assertEqual(b'original', marker.read_bytes())

  def test_preexisting_symlink_root_is_not_followed(self):
    outside = self.dataset.parent / 'outside'
    outside.mkdir()
    try:
      (self.dataset / 'flower_photos').symlink_to(outside, target_is_directory=True)
    except OSError:
      self.skipTest('Symlink creation is unavailable on this platform.')
    self.assertRejected(self.archive([self.file('flower_photos/file')]))
    self.assertEqual([], os.listdir(str(outside)))

  def test_destination_appearing_during_download_is_preserved(self):
    root = self.dataset / 'flower_photos'
    rename = os.rename

    def insert_existing_root(source, destination):
      # Simulate a nonempty destination racing with the promotion syscall.
      root.mkdir()
      (root / 'original').write_bytes(b'unchanged')
      return rename(source, destination)

    contents = self.archive([self.file('flower_photos/file')])
    with mock.patch.object(dataset_download.os, 'rename',
                           side_effect=insert_existing_root):
      with self.assertRaises(OSError):
        self.download(contents)
    self.assertEqual(b'unchanged', (root / 'original').read_bytes())
    self.assertEqual(['flower_photos'], os.listdir(str(self.dataset)))

  def test_promotion_error_cleans_staging(self):
    with mock.patch.object(dataset_download.os, 'rename',
                           side_effect=OSError('promotion failed')):
      self.assertRejected(self.archive([self.file('flower_photos/file')]))

  def test_member_count_and_expanded_size_limits(self):
    with mock.patch.object(dataset_download, '_MAX_MEMBERS', 1):
      self.assertRejected(self.archive([
          self.file('flower_photos/a'), self.file('flower_photos/b')]))
    with mock.patch.object(dataset_download, '_MAX_EXTRACTED_BYTES', 1):
      self.assertRejected(self.archive([self.file('flower_photos/file')]))

  def test_excessive_path_depth_is_rejected_and_cleaned(self):
    path = '/'.join(['flower_photos'] + ['nested'] * 64 + ['file'])
    self.assertRejected(self.archive([self.file(path)]))

  def test_download_size_limit(self):
    with mock.patch.object(dataset_download, '_MAX_DOWNLOAD_BYTES', 1):
      self.assertRejected(self.archive([self.file('flower_photos/file')]))

  def test_download_error_cleans_staging(self):
    opener = mock.Mock()
    opener.open.side_effect = OSError('download failed')
    with mock.patch.object(dataset_download.request, 'build_opener',
                           return_value=opener):
      with self.assertRaises(OSError):
        dataset_download.download_and_uncompress_tarball(
            'https://example.test/data.tgz', str(self.dataset), '0' * 64,
            'flower_photos')
    self.assertEqual([], os.listdir(str(self.dataset)))

  def test_empty_or_root_file_archive_is_rejected(self):
    self.assertRejected(self.archive([]))
    self.assertRejected(self.archive([self.file('flower_photos')]))

  def test_invalid_root_is_rejected_before_download(self):
    for root in ('../flower_photos', '/flower_photos', 'flower_photos/nested', ''):
      with self.subTest(root=root), mock.patch.object(
          dataset_download.request, 'build_opener') as opener:
        self.assertRejected(b'archive', root=root)
        opener.assert_not_called()


if __name__ == '__main__':
  unittest.main()
