"""
This script inputs 204-class-long label vectors and outputs 57-class-long
label vectors that are the product of consolidating or removing classes. One
of three operations will be peformed per class 'position': 1) copy, 2) bitwise
or and copy, or 3) don't copy. The logical or will take in a list of the '
positions in the source vector to aggregate into a given position in the
destination vector.

ORIGINAL FILES WILL BE OVERWRITTEN
"""

import argparse as ap
import numpy as np
import os

parser = ap.ArgumentParser()

parser.add_argument('--label_vector_dir_path', required=True)

args = parser.parse_args()

num_classes = 57

map_vector = np.array(
  [0, 1, 2, 3, 4, 5, 6, 10, 7, 10, 8, 11, 9, 11, 8, 11, 9, 11, 6, 10, 7, 10, 12,
   14, 13, 14, 12, 14, 13, 14, 0, 15, 17, 0, 16, 19, 0, 15, 18, 0, 16, 20, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 23, 0, 22, 25, 0, 21, 24, 0, 22, 26, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, 27, 29, 31, 33, 28, 30, 31, 34, 37, 27, 29,
   32, 35, 28, 30, 32, 36, 38, 27, 29, 31, 33, 28, 30, 31, 34, 37, 27, 29, 32,
   35, 28, 30, 32, 36, 38, 27, 29, 31, 33, 28, 30, 31, 34, 37, 27, 29, 32, 35,
   28, 30, 32, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
   53, 54, 55, 56], dtype=np.int32)

for file_name in os.listdir(args.label_vector_dir_path):
  label_vector_file_path = os.path.join(args.label_vector_dir_path, file_name)

  input_vector = np.load(label_vector_file_path)

  output_vector = np.zeros((num_classes,), dtype=np.uint8)

  for i in range(num_classes):
    output_vector[i] = np.any(input_vector[map_vector == i]).astype(np.uint8)

  np.save(label_vector_file_path, output_vector)