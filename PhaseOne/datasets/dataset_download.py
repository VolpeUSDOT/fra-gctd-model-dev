"""Verified HTTPS downloads and restricted extraction for dataset archives."""

import hashlib
import ntpath
import os
import re
import shutil
import ssl
import tarfile
import tempfile
from urllib import parse, request


_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
_MAX_MEMBERS = 10000
_MAX_PATH_COMPONENTS = 64
_WINDOWS_RESERVED_NAMES = {'CON', 'PRN', 'AUX', 'NUL'} | {
    '{}{}'.format(prefix, number)
    for prefix in ('COM', 'LPT') for number in range(1, 10)}


class _HTTPSOnlyRedirectHandler(request.HTTPRedirectHandler):

  def redirect_request(self, req, fp, code, msg, headers, newurl):
    url = parse.urlsplit(newurl)
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
      raise ValueError('Dataset download redirects must use HTTPS.')
    return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_and_uncompress_tarball(tarball_url, dataset_dir, sha256,
                                    archive_root):
  """Verifies an archive and promotes its single dataset directory atomically.

  Only relative regular files and directories beneath archive_root are allowed.
  Existing dataset roots are refused, including symlinks. This implementation
  does not use tarfile extraction filters and supports Python 3.7 and newer.
  """
  url = parse.urlsplit(tarball_url)
  if url.scheme != 'https' or not url.hostname or url.username or url.password:
    raise ValueError('Dataset downloads require an HTTPS URL without credentials.')
  if not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
    raise ValueError('A trusted SHA-256 digest is required.')
  if not re.fullmatch(r'[A-Za-z0-9_-]+', archive_root):
    raise ValueError('Invalid dataset archive root.')

  dataset_dir = os.path.realpath(dataset_dir)
  os.makedirs(dataset_dir, exist_ok=True)
  destination = os.path.join(dataset_dir, archive_root)
  if os.path.lexists(destination):
    raise ValueError('Dataset root already exists: {}'.format(destination))

  opener = request.build_opener(
      request.HTTPSHandler(context=ssl.create_default_context()),
      _HTTPSOnlyRedirectHandler())
  with tempfile.TemporaryDirectory(prefix='.dataset-download-',
                                   dir=dataset_dir) as staging:
    archive_path = os.path.join(staging, 'archive.tar.gz')
    digest = hashlib.sha256()
    downloaded = 0
    with opener.open(tarball_url, timeout=30) as response, \
        open(archive_path, 'xb') as output:
      if parse.urlsplit(response.geturl()).scheme != 'https':
        raise ValueError('Dataset download response must use HTTPS.')
      while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
          break
        downloaded += len(chunk)
        if downloaded > _MAX_DOWNLOAD_BYTES:
          raise ValueError('Dataset download exceeds the size limit.')
        digest.update(chunk)
        output.write(chunk)
    if digest.hexdigest() != sha256:
      raise ValueError('Dataset archive SHA-256 mismatch.')

    extracted = os.path.join(staging, 'extracted')
    os.mkdir(extracted)
    seen = set()
    expanded = 0
    # Copy data ourselves: never apply archive links, owners, or permissions.
    with tarfile.open(archive_path, 'r|gz') as archive:
      for member in archive:
        name = member.name.rstrip('/') if member.isdir() else member.name
        parts = name.split('/')
        if (parts[0] != archive_root or ntpath.isabs(name) or
            len(parts) > _MAX_PATH_COMPONENTS or
            any(part in ('', '.', '..') or part.endswith(('.', ' ')) or
                any(char in part for char in '\\:*?"<>|') or
                any(ord(char) < 32 for char in part) or
                part.split('.')[0].upper() in _WINDOWS_RESERVED_NAMES
                for part in parts)):
          raise ValueError('Unsafe dataset archive path: {}'.format(member.name))
        target = os.path.realpath(os.path.join(extracted, *parts))
        if os.path.commonpath([extracted, target]) != extracted:
          raise ValueError('Archive member escapes the extraction root.')
        if not (member.isdir() or member.isreg()) or member.issparse():
          raise ValueError('Only regular files and directories are allowed.')
        if name in seen or len(seen) >= _MAX_MEMBERS:
          raise ValueError('Duplicate archive member or member limit exceeded.')
        seen.add(name)
        expanded += member.size
        if member.size < 0 or expanded > _MAX_EXTRACTED_BYTES:
          raise ValueError('Dataset archive exceeds the extracted size limit.')
        if member.isdir():
          os.makedirs(target, exist_ok=True)
        else:
          os.makedirs(os.path.dirname(target), exist_ok=True)
          with archive.extractfile(member) as source, open(target, 'xb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)

    staged_root = os.path.join(extracted, archive_root)
    if not os.path.isdir(staged_root):
      raise ValueError('Dataset archive does not contain its expected directory.')
    if os.path.lexists(destination):
      raise ValueError('Dataset root appeared during download.')
    os.rename(staged_root, destination)
