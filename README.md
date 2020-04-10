### Dependencies
We use python 3 for this project.

Make sure `FFMPEG` is installed, or install by
`sudo apt install ffmpeg`

Install the python packages dependencies by
`pip install -r requirements.txt`


### Preparing Data
Download the data from Total Capture official [site](https://cvssp.org/projects/totalcapture/TotalCapture/).

> Please Note that we have **NO** permission to redistribute this dataset. Please never ask us for a copy.

Then organize the data in below structure (put all data in `./data/totalcapture`):

```
./data/totalcapture
├── calibration.cal
├── s1
│   ├── acting1
│   │   ├── acting1_BlenderZXY_YmZ.bvh
│   │   ├── acting1_Xsens_AuxFields.sensors
│   │   ├── gt_skel_gbl_ori.txt
│   │   ├── gt_skel_gbl_pos.txt
│   │   ├── s1_acting1_calib_imu_bone.txt
│   │   ├── s1_acting1_calib_imu_ref.txt
│   │   ├── s1_acting1_Xsens.sensors
│   │   ├── TC_S1_acting1_cam1.mp4
│   │   ├── TC_S1_acting1_cam2.mp4
│   │   ├── TC_S1_acting1_cam3.mp4
│   │   ├── TC_S1_acting1_cam4.mp4
│   │   ├── TC_S1_acting1_cam5.mp4
│   │   ├── TC_S1_acting1_cam6.mp4
│   │   ├── TC_S1_acting1_cam7.mp4
│   │   └── TC_S1_acting1_cam8.mp4
│   ├── acting2
│   │   ├── ...
│   ├── ...
├── s2
...
```

### Usage

```
python gendata/gendb.py

# if cannot import tools
# export PYTHONPATH=".:$PYTHONPATH"; python gendata/gendb.py  
```

**Options**: 
You can change options in `gendata/config.yaml`, e.g.

- save_frame: whether extract frames from the videos
- gen_train: generate dataset for training
- gen_test: generate dataset for testing

Finally, you should have a bunch of images and two `.pkl` files in the `./data/images`. Move the `.pkl` files to `../annot` directory, and you get

```
./data/
├── images
│   ├── s_01_act_01_subact_01_ca_01
│   │   ├── 000000.jpg
│   │   ├── ...
│   ├── ...
├── annot
│   ├── totalcapture_train.pkl
│   ├── totalcapture_validation.pkl
```

If you want to archive all images in one zip file which is more efficient to be transferred to server or cloud storage, make sure your `pwd` is parent directory of `images/`, then run `zip -0 -r images.zip images/`.

