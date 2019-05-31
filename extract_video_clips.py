"""
python extract_video_clips.py -i C:/Users/Public/fra-gctd-project/Raw_Data/ramsey_nj/20180419/Ch02_20180419000000_20180419235959_12.avi
"""
import argparse as ap
import logging
import numpy as np
from os import environ, path, makedirs
from subprocess import PIPE, Popen, TimeoutExpired

parser = ap.ArgumentParser()

parser.add_argument('--inputvideofilepath', '-i', required=True)
parser.add_argument('--outputclipdirpath', '-o', default=None)
parser.add_argument('--highresoutputdirname', '-hn', default='high_res')
parser.add_argument('--lowresoutputdirname', '-ln', default='low_res')
parser.add_argument('--cliplength', '-cl', type=int, default=64)
parser.add_argument('--inputheight', '-ih', type=int, default=1080)
parser.add_argument('--inputwidth', '-iw', type=int, default=1920)
parser.add_argument('--numchannels', '-nc', type=int, default=3)
parser.add_argument('--crop', '-c', action='store_true')
parser.add_argument('--cropheight', '-ch', type=int, default=920,
                    help='y-component of bottom-right corner of crop.')
parser.add_argument('--cropwidth', '-cw', type=int, default=1920,
                    help='x-component of bottom-right corner of crop.')
parser.add_argument('--cropx', '-cx', type=int, default=0,
                    help='x-component of top-left corner of crop.')
parser.add_argument('--cropy', '-cy', type=int, default=82,
                    help='y-component of top-left corner of crop.')
parser.add_argument('--scaleheight', '-sh', type=int, default=224)
parser.add_argument('--scalewidth', '-sw', type=int, default=224)

args = parser.parse_args()

video_file_name = path.splitext(path.basename(args.inputvideofilepath))[0]

if args.outputclipdirpath is None:
  outputclipdirpath = path.join(path.dirname(args.inputvideofilepath), video_file_name)
else:
  outputclipdirpath = path.join(args.outputclipdirpath, video_file_name)

high_res_clip_dir_path = path.join(outputclipdirpath, args.highresoutputdirname)

low_res_clip_dir_path = path.join(outputclipdirpath, args.lowresoutputdirname)

if not path.isdir(high_res_clip_dir_path):
  makedirs(high_res_clip_dir_path)

if not path.isdir(low_res_clip_dir_path):
  makedirs(low_res_clip_dir_path)

try:
  ffmpeg_path = environ['FFMPEG_PATH']
except KeyError:
  logging.warning('Environment variable FFMPEG_PATH not set. Attempting '
                  'to use default ffmpeg binary location.')
  ffmpeg_path = '/usr/local/bin/ffmpeg'

  if not path.exists(ffmpeg_path):
    ffmpeg_path = '/usr/bin/ffmpeg'

input_ffmpeg_command = [ffmpeg_path, '-i', args.inputvideofilepath]

input_ffmpeg_command.extend(
  ['-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-vsync', 'vfr', '-vf',
   'crop={}:{}:{}:{}'.format(
     args.cropwidth, args.cropheight, args.cropx, args.cropy),
   '-hide_banner', '-loglevel', '0', '-f', 'image2pipe', 'pipe:1'])

clip_string_len = \
  args.cliplength * args.cropheight * args.cropwidth * args.numchannels

buffer_scale = 2

# multiply by two since we are extracting double the clip length and creating
# two clips from the even/odd-stratified samples
while buffer_scale < clip_string_len * 2:
  buffer_scale *= 2

low_res_output_clip_string_len = \
  args.cliplength * args.scaleheight * args.scalewidth * args.numchannels

low_res_output_buffer_scale = 2

# multiply by two since we are extracting double the clip length and creating
# two clips from the even/odd-stratified samples
while low_res_output_buffer_scale < low_res_output_clip_string_len * 2:
  low_res_output_buffer_scale *= 2

frame_pipe = Popen(
  input_ffmpeg_command, stdout=PIPE, stderr=PIPE, bufsize=buffer_scale)

previous_frame_array = None

high_res_output_ffmpeg_command_prefix = [
  ffmpeg_path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '{}x{}'.format(
    args.cropwidth, args.cropheight), '-i', 'pipe:', '-c:v', 'libx264',
  '-framerate', '29.830732', '-vsync', 'vfr']

low_res_output_ffmpeg_command_prefix = [
  ffmpeg_path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '{}x{}'.format(
    args.cropwidth, args.cropheight), '-i', 'pipe:', '-c:v', 'libx264',
  '-framerate', '29.830732', '-vsync', 'vfr', '-vf', 'scale={}:{}'.format(
    args.scalewidth, args.scaleheight)]

clip_number = 0

while True:
  frame_string = frame_pipe.stdout.read(clip_string_len * 2)
  print('frame_string_len: {}'.format(len(frame_string)))

  if frame_string is None or len(frame_string) == 0:
    frame_pipe.stdout.close()
    frame_pipe.stderr.close()
    frame_pipe.terminate()

    previous_frame_array = None

    print('killed frame_pipe')

    break

  frame_array = np.fromstring(frame_string, dtype=np.uint8)
  print('frame_array_shape: {}'.format(frame_array.shape))

  frame_array = np.reshape(
    frame_array, 
    [-1, args.cropheight, args.cropwidth, args.numchannels])
  print('frame_array_shape: {}'.format(frame_array.shape))
  
  # if the last clip is short, repeat the last frame
  if frame_array.shape[0] < args.cliplength * 2:
    frame_array = np.concatenate((frame_array, np.tile(
      frame_array[-1], 
      (args.cliplength * 2 - frame_array.shape[0], 1, 1, 1))))
    print('frame_array_shape: {}'.format(frame_array.shape))

  # write frame_array to new video clip
  if previous_frame_array is not None:
    # here, rather than divide by two, we simply do not multiply by
    intermediate_frame_array = np.concatenate(
      (previous_frame_array[args.cliplength:],
       frame_array[:args.cliplength]), axis=0)

    array_list = [intermediate_frame_array[0::2, :, :, :],
                  intermediate_frame_array[1::2, :, :, :]]

    for array in array_list:
      array = np.reshape(array, [
        args.cliplength * args.cropheight * args.cropwidth * args.numchannels])

      # write intermediate_frame_array to new video clip
      intermediate_frame_string = array.tostring()

      high_res_output_ffmpeg_command = \
        high_res_output_ffmpeg_command_prefix + [
        path.join(high_res_clip_dir_path, '{}_{:07d}.avi'.format(
          video_file_name, clip_number))]

      high_res_output_frame_pipe = Popen(
        high_res_output_ffmpeg_command, stdin=PIPE, stderr=PIPE,
        bufsize=buffer_scale)

      try:
        outs, errs = high_res_output_frame_pipe.communicate(
          intermediate_frame_string, timeout=15)
        print('intermediate_outs: {}'.format(outs))
        print('intermediate_errs: {}'.format(errs))
      except TimeoutExpired:
        high_res_output_frame_pipe.kill()
        print('killed intermediate_output_frame_pipe')

      low_res_output_ffmpeg_command = low_res_output_ffmpeg_command_prefix + [
        path.join(low_res_clip_dir_path, '{}_{:07d}.avi'.format(
          video_file_name, clip_number))]

      low_res_output_frame_pipe = Popen(
        low_res_output_ffmpeg_command, stdin=PIPE, stderr=PIPE,
        bufsize=low_res_output_buffer_scale)

      try:
        outs, errs = low_res_output_frame_pipe.communicate(
          intermediate_frame_string, timeout=15)
        print('intermediate_outs: {}'.format(outs))
        print('intermediate_errs: {}'.format(errs))
      except TimeoutExpired:
        low_res_output_frame_pipe.kill()
        print('killed intermediate_output_frame_pipe')

      clip_number += 1

  previous_frame_array = frame_array

  array_list = [frame_array[0::2, :, :, :],
                frame_array[1::2, :, :, :]]

  for array in array_list:
    array = np.reshape(array, [
      args.cliplength * args.cropheight * args.cropwidth * args.numchannels])
    print('array_shape: {}'.format(array.shape))

    frame_string = array.tostring()
    print('string_len: {}'.format(len(frame_string)))

    high_res_output_ffmpeg_command = high_res_output_ffmpeg_command_prefix + [
      path.join(high_res_clip_dir_path, '{}_{:07d}.avi'.format(
        video_file_name, clip_number))]

    high_res_output_frame_pipe = Popen(
      high_res_output_ffmpeg_command, stdin=PIPE, stderr=PIPE,
      bufsize=buffer_scale)

    try:
      outs, errs = high_res_output_frame_pipe.communicate(
        frame_string, timeout=15)
      print('outs: {}'.format(outs))
      print('errs: {}'.format(errs))
    except TimeoutExpired:
      high_res_output_frame_pipe.kill()
      print('killed output_frame_pipe')

    low_res_output_ffmpeg_command = low_res_output_ffmpeg_command_prefix + [
      path.join(low_res_clip_dir_path, '{}_{:07d}.avi'.format(
        video_file_name, clip_number))]

    low_res_output_frame_pipe = Popen(
      low_res_output_ffmpeg_command, stdin=PIPE, stderr=PIPE,
      bufsize=low_res_output_buffer_scale)

    try:
      outs, errs = low_res_output_frame_pipe.communicate(
        frame_string, timeout=15)
      print('outs: {}'.format(outs))
      print('errs: {}'.format(errs))
    except TimeoutExpired:
      low_res_output_frame_pipe.kill()
      print('killed output_frame_pipe')

    clip_number += 1

