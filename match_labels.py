import argparse as ap
import numpy as np
import os

"""
python match_labels.py --data_source_path /media/data_0/fra/gctd/Data_Sources/ramsey_nj/seed_data_source
"""

parser = ap.ArgumentParser()

parser.add_argument('--data_source_path', 
                    default='C:/Users/Public/fra-gctd-project/Data_Sources/'
                            'ramsey_nj/seed_data_source')

args = parser.parse_args()

labels_dir_path = os.path.join(args.data_source_path, 'labels')
label_file_paths = [os.path.join(labels_dir_path, label_file_name)
                    for label_file_name in sorted(os.listdir(labels_dir_path))]

num_mismatches = 0

for i in range(0, len(label_file_paths), 2):
  label_0 = np.load(label_file_paths[i])
  label_1 = np.load(label_file_paths[i+1])

  where_equal = np.equal(label_0, label_1)
  all_equal = np.all(where_equal)

  if not all_equal:
    num_mismatches += 1

    print('Index: {}, Name: {}, Mismatches: ({})'.format(
      i, os.path.basename(label_file_paths[i]),
      np.squeeze(np.argwhere(np.bitwise_not(where_equal)))))

print('Total mismatches: {}'.format(num_mismatches))