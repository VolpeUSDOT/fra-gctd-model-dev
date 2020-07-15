"""
This script inputs 204-class-long label vectors and outputs 96-class-long
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

num_classes = 96

map_vector = np.array(
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
   22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
   41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
   60, 61, 62, 63, 64, 65, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
   -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 66, 68, 70, 72,
   67, 69, 70, 73, 76, 66, 68, 71, 74, 67, 69, 71, 75, 77, 66, 68, 70, 72, 67,
   69, 70, 73, 76, 66, 68, 71, 74, 67, 69, 71, 75, 77, 66, 68, 70, 72, 67, 69,
   70, 73, 76, 66, 68, 71, 74, 67, 69, 71, 75, 77, 78, 79, 80, 81, 82, 83, 84,
   85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95], dtype=np.int32)

for file_name in os.listdir(args.label_vector_dir_path):
  label_vector_file_path = os.path.join(args.label_vector_dir_path, file_name)

  input_vector = np.load(label_vector_file_path)

  output_vector = np.zeros((num_classes,), dtype=np.uint8)

  for i in range(num_classes):
    output_vector[i] = np.any(input_vector[map_vector == i]).astype(np.uint8)

  np.save(label_vector_file_path, output_vector)