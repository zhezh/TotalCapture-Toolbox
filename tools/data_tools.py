import os
from collections import deque
import re
import numpy as np
from pyquaternion import Quaternion
import bvh


bone_names = ('Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
              'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot')  # imu bones

# marker name of vicon
vicon_joints = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder',
                'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
                'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot')

bvh_joints = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head',
              'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'RightHandEnd', 'RightHandThumb1',
              'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'LeftHandEnd', 'LeftHandThumb1',
              'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
              'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase')

# should check bone start and end order, e.g. neck to head or vice versa
imu_bone_vicon_start = {'Head': 'Neck', 'Sternum': 'Spine3', 'Pelvis': 'Spine',  # not sure about this row
                        'L_UpArm': 'LeftArm', 'R_UpArm': 'RightArm',
                        'L_LowArm': 'LeftForeArm', 'R_LowArm': 'RightForeArm',
                        'L_UpLeg': 'LeftUpLeg', 'R_UpLeg': 'RightUpLeg',
                        'L_LowLeg': 'LeftLeg', 'R_LowLeg': 'RightLeg',
                        'L_Foot': 'LeftFoot', 'R_Foot': 'RightFoot'}

imu_bone_vicon_end   = {'Head': 'Head', 'Sternum': 'Neck', 'Pelvis': 'Spine1',  # not sure about this row
                        'L_UpArm': 'LeftForeArm', 'R_UpArm': 'RightForeArm',
                        'L_LowArm': 'LeftHand', 'R_LowArm': 'RightHand',
                        'L_UpLeg': 'LeftLeg', 'R_UpLeg': 'RightLeg',
                        'L_LowLeg': 'LeftFoot', 'R_LowLeg': 'RightFoot',
                        'L_Foot': 'LeftToeBase', 'R_Foot': 'RightToeBase'}

# re
float_reg = r'(\-?\d+\.\d+)'
int_reg = r'[\d]+'
word_reg = r'[a-zA-z\_]+'

float_finder = re.compile(float_reg)
int_finder = re.compile(int_reg)
word_finder = re.compile(word_reg)


def parse_sensor_6axis(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)

    seq_meta = lines.popleft()
    seq_meta_result = int_finder.findall(seq_meta)
    assert len(seq_meta_result) == 2, 'error in seq meta data'
    num_sensors = int(seq_meta_result[0])
    num_frames = int(seq_meta_result[1])

    frames = []
    for f in range(num_frames):
        # frame
        frame_index_line = lines.popleft()
        frame_index = int(int_finder.findall(frame_index_line)[0])
        joints = dict()
        joints['index'] = frame_index  # frame index
        # joint
        for i in range(num_sensors):
            onejoint = lines.popleft()
            joint_name = word_finder.findall(onejoint)[0]
            assert joint_name in bone_names, 'invalid joint name: {} in frame {}'.format(joint_name, frame_index)
            values = float_finder.findall(onejoint)
            assert len(values) == 7, 'wrong number of joint parameter'
            orientation = tuple(float(x) for x in values[0:4])
            acceleration = tuple(float(x) for x in values[4:7])
            joints[joint_name] = (orientation, acceleration)

        frames.append(joints)
    return frames


def parse_calib_imu_ref(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    ref_num_sensors = int(lines.popleft())
    assert ref_num_sensors == len(bone_names), 'mismatching sensor nums with ref sensor nums'
    ref_joints = dict()
    for i in range(ref_num_sensors):
        onejoint = lines.popleft()
        joint_name = word_finder.findall(onejoint)[0]
        assert joint_name in bone_names, 'invalid joint name: {}'.format(joint_name)
        values = float_finder.findall(onejoint)
        assert len(values) == 4, 'wrong number of joint parameter'
        # orientation in ref is ordered as (x y z w), which is different from captured data (w x y z)
        orientation_imag = [float(x) for x in values[0:3]]
        orientation = [float(values[3])]
        orientation.extend(orientation_imag)
        ref_joints[joint_name] = orientation
    return ref_joints


def parse_calib_imu_bone(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    bone_num_sensors = int(lines.popleft())
    assert bone_num_sensors == len(bone_names), 'mismatching sensor nums with ref sensor nums'
    ref_bones = dict()
    for i in range(bone_num_sensors):
        onejoint = lines.popleft()
        joint_name = word_finder.findall(onejoint)[0]
        assert joint_name in bone_names, 'invalid joint name: {}'.format(joint_name)
        values = float_finder.findall(onejoint)
        assert len(values) == 4, 'wrong number of joint parameter'
        # orientation in ref is ordered as (x y z w), which is different from captured data (w x y z)
        orientation_imag = [float(x) for x in values[0:3]]
        orientation = [float(values[3])]
        orientation.extend(orientation_imag)
        ref_bones[joint_name] = orientation
    return ref_bones


def parse_vicon_gt_ori(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    joints = lines[0].split()
    assert joints == list(vicon_joints), 'mismatching joint names with vicon gt'

    ori_frames = []
    for i in range(1,len(lines)):
        oneline = lines[i]
        vals = list(map(float, oneline.split()))
        if len(vals) == 0:
            break  # in case empty line at the file end
        assert len(vals) == 4*len(joints), 'oops, mismatching joint nums and orientation data'
        joint_ori_frame = dict()
        for j in range(len(joints)):
            # quaternion order: xyzw, should do order manipulation
            joint_ori_frame[joints[j]] = [vals[4*j+3], vals[4*j], vals[4*j+1], vals[4*j+2]]
        ori_frames.append(joint_ori_frame)
    return ori_frames


def parse_vicon_gt_pos(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    joints = lines[0].split()
    assert joints == list(vicon_joints), 'mismatching joint names with vicon gt'
    pos_frames = []
    for i in range(1,len(lines)):
        oneline = lines[i]
        vals = list(map(float, oneline.split()))
        if len(vals) == 0:
            break  # in case empty line at the file end
        assert len(vals) == 3*len(joints), 'oops, mismatching joint nums and position data'
        joint_pos_frame = dict()
        for j in range(len(joints)):
            joint_pos_frame[joints[j]] = vals[3*j:3*(j+1)]
        pos_frames.append(joint_pos_frame)
    return pos_frames


def parse_imu_bone_info(fpath):
    with open(fpath, 'r') as f:
        mocap = bvh.Bvh(f.read())
    all_joints = mocap.get_joints()
    joints_names = mocap.get_joints_names()

    bone_info = dict()
    for b in bone_names:
        start = imu_bone_vicon_start[b]
        end = imu_bone_vicon_end[b]
        if start == end:
            this_joint = all_joints[mocap.get_joint_index(start)]
            bone_length = tuple(float(x) for x in this_joint.children[2].children[0].value[1:])
        else:
            bone_length = mocap.joint_offset(end)
        bone_info[b] = (start, end, bone_length)
    return bone_info


def parse_camera_cal(fpath):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = deque(lines)
    num_cameras, distortion_order = lines.popleft().split()
    num_cameras = int(num_cameras)
    distortion_order = int(distortion_order)
    cameras = []
    for i in range(num_cameras):
        min_row, max_row, min_col, max_col = tuple(map(int, lines.popleft().split()))
        fx, fy, cx, cy = tuple(map(float, lines.popleft().split()))
        distor_param = float(lines.popleft())
        r1 = list(map(float, lines.popleft().split()))
        r2 = list(map(float, lines.popleft().split()))
        r3 = list(map(float, lines.popleft().split()))
        R = np.array([r1, r2, r3])
        t = np.reshape(np.array(list(map(float, lines.popleft().split()))), (3,1))
        cam = {'R': R, 'T': t, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'distor': distor_param}
        cameras.append(cam)
    return cameras
