import argparse
import json
import numpy as np
import os
from subprocess import PIPE, run
import tensorflow as tf

"""
python construct_dataset.py --data_source_dir_path /media/data_0/fra/gctd/Data_Sources/ramsey_nj/seed_data_source --data_set_dir_path /media/data_0/fra/gctd/Data_Sets/ramsey_nj/seed_data_set
"""
parser = argparse.ArgumentParser()

parser.add_argument('--data_source_dir_path', required=True)
parser.add_argument('--data_set_dir_path', required=True)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--clip_length', default=64, type=int)

args = parser.parse_args()

if not os.path.exists(args.data_source_dir_path):
  raise ValueError("data_source_dir_path: {} doe not exist.".format(args.data_source_dir_path))

if not os.path.exists(args.data_set_dir_path):
  os.makedirs(args.data_set_dir_path)

num_samples = len(os.listdir(os.path.join(args.data_source_dir_path, 'labels')))

try:
  ffmpeg_path = os.environ['FFMPEG_PATH']
except KeyError:
  ffmpeg_path = '/usr/local/bin/ffmpeg'

  if not os.path.exists(ffmpeg_path):
    ffmpeg_path = '/usr/bin/ffmpeg'

ffmpeg_command_prefix = [ffmpeg_path, '-i']

ffmpeg_command_suffix = [
  '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr',
  '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1']

try:
  ffprobe_path = os.environ['FFPROBE_PATH']
except KeyError:
  ffprobe_path = '/usr/local/bin/ffprobe'

  if not os.path.exists(ffprobe_path):
    ffprobe_path = '/usr/bin/ffprobe'

input_ffprobe_command_prefix = [ffprobe_path, '-show_streams', '-print_format',
                                'json', '-loglevel', 'warning']

scale_height, scale_width = 224, 224

clip_string_len = args.clip_length * scale_height * scale_width * 3

def invoke_subprocess(command):
  completed_subprocess = run(command, stdout=PIPE, stderr=PIPE, timeout=60)
  if len(completed_subprocess.stderr) > 0:
    std_err = str(completed_subprocess.stderr, encoding='utf-8')
    raise Exception(std_err)
  return completed_subprocess.stdout

def get_video_dimensions(video_file_path):
  command = input_ffprobe_command_prefix + [video_file_path]
  output = invoke_subprocess(command)
  json_map = json.loads(str(output, encoding='utf-8'))
  return int(json_map['streams'][0]['height']), \
         int(json_map['streams'][0]['width'])

def get_video_clip(video_file_path):
  command = ffmpeg_command_prefix + [video_file_path] + ffmpeg_command_suffix
  output = invoke_subprocess(command)
  return output

def gen_fn():
  features_dir_path = os.path.join(args.data_source_dir_path, 'low_res')
  feature_file_paths = [os.path.join(features_dir_path, file_name)
                        for file_name in os.listdir(features_dir_path)]
  labels_dir_path = os.path.join(args.data_source_dir_path, 'labels')

  for feature_file_path in feature_file_paths:
    feature_string = get_video_clip(feature_file_path)
    feature_array = np.frombuffer(feature_string, dtype=np.uint8)

    label_array = np.load(os.path.join(labels_dir_path, os.path.splitext(
      os.path.basename(feature_file_path))[0] + '.npy'))

    example = {
      'feature': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[feature_array.tostring()])),
      'label': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[label_array.tostring()]))
    }

    yield tf.train.Example(
      features=tf.train.Features(feature=example)).SerializeToString()

dataset = tf.data.Dataset.from_generator(
  gen_fn, output_types=tf.string, output_shapes=[])

dataset = dataset.shuffle(num_samples).batch(args.batch_size)

initializer = dataset.make_one_shot_iterator()

next_batch = initializer.get_next()

example = tf.data.Dataset.from_tensor_slices(next_batch)

file_name = tf.placeholder(dtype=tf.string)

writer = tf.data.experimental.TFRecordWriter(file_name)

write_dataset = writer.write(example)

batch_num = 0

with tf.Session().as_default() as sess:
  while True:
    try:
      filename = os.path.join(args.data_set_dir_path, '{:07d}.tfrecord'.format(
        batch_num))
      sess.run(write_dataset, feed_dict={file_name: filename})
      batch_num += 1
    except tf.errors.OutOfRangeError:
      break