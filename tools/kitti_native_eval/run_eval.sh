#!/bin/bash

# 1 kitti_native_eval_root
# 2 label_dir
# 3 score_threshold
# 4 epoch
# 5 results_path

set -e

cd $1
echo "$4" | tee -a $5
./evaluate_object_3d_offline $2 $3/$4 | tee -a ./$5
