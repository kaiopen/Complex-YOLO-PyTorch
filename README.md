# Complex-YOLO
An unofficial implementation of [Complex-YOLO: An Euler-Region-Proposal for Real-time 3D Object Detection on Point Clouds](https://arxiv.org/abs/1803.06199) based on [AI-liu/Complex-YOLO](https://github.com/AI-liu/Complex-YOLO) and [Yuanchu/YOLO3D](https://github.com/Yuanchu/YOLO3D).

## Data
The tree of datasets directory is as follows.

 Kitti<br>
 ├── training<br>
 │   ├── calib<br>
 │   ├── image_2<br>
 │   ├── label_2<br>
 │   ├── planes<br>
 │   └── velodyne<br>
 ├── train.txt<br>
 ├── trainval.txt<br>
 └── val.txt<br>

## Preparing
Before training or testing, bird's eye view maps should be generated.
```
python tools/get_bevs_lidar.py
```
The BEV maps will be saved in `datasets_cache`.

## Training
```
python tools/train.py --use_cuda --pin_memory
```
The defualt batch size is 12. You can change it if memory is not enough with `--batch_size`.

## Testing
```
python tools/test.py --use_cuda --split="val"
```
The result will be shown as pictures.