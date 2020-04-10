import os
import sys
sys.path.insert(0, '.')
import numpy as np
import pickle
import cv2
import yaml
import copy
import tools.data_tools as dt
import skvideo
# skvideo.setFFmpegPath('../FFmpeg/')  # manually compiled ffmpeg
from skvideo.io import vreader, ffprobe, FFmpegReader
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from tqdm import tqdm
plt.ioff()


def get_train_val_dirs(config):
    train_dirs = []
    test_dirs = []

    for s in config['train_subs']:
        sub_path = 's{}'.format(s)
        for a in config['train_seqs']:
            for i in config['train_seqs'][a]:
                action_path = '{}{}'.format(a, i)
                seq_dir = os.path.join(config['dataset_root_dir'], sub_path, action_path)
                meta = {
                    'subject': s,
                    'action': config['action_map'][a],
                    'subaction': i}
                train_dirs.append((seq_dir, meta))

    for s in config['test_subs']:
        sub_path = 's{}'.format(s)
        for a in config['test_seqs']:
            for i in config['test_seqs'][a]:
                action_path = '{}{}'.format(a, i)
                seq_dir = os.path.join(config['dataset_root_dir'], sub_path, action_path)
                meta = {
                    'subject': s,
                    'action': config['action_map'][a],
                    'subaction': i}
                test_dirs.append((seq_dir, meta))

    return train_dirs, test_dirs


def load_camera(config):
    cal_file = os.path.join(config['dataset_root_dir'], 'calibration.cal')
    cameras = dt.parse_camera_cal(cal_file)
    cam_idx = 1
    for cam in cameras:
        intrinsic_mat = np.array([[cam['fx'], 0., cam['cx']],
                                  [0., cam['fy'], cam['cy']],
                                  [0., 0., 1.]])
        extrinsic_mat = np.concatenate((cam['R'], np.reshape(cam['T'], (-1,1))), axis=1)
        cam['intri_mat'] = intrinsic_mat
        cam['extri_mat'] = extrinsic_mat
        cam['k'] = np.zeros([3,1])  # make cam compatible
        cam['k'][0,0] = cam['distor']  # totalcapture only use 1-order radial distortion
        cam['p'] = np.zeros([2,1])
        cam['name'] = 'cam{}'.format(cam_idx)
        cam_idx += 1
    return cameras


def extract_db(config, dir_meta, cameras):
    """

    :param config: config from config.yaml
    :param dir_meta: determine extraction of train or test
    :param cameras:
    :return:
    """
    dataset = []  # all images of train or test
    all_joints = dt.vicon_joints
    for dir, meta in dir_meta:  # one action contains 8 cam views
        meta_sub = meta['subject']
        meta_act = config['action_reverse_map'][meta['action']]  # action string name
        meta_subact = meta['subaction']

        gt_pos_path = os.path.join(dir, 'gt_skel_gbl_pos.txt')
        gt_ori_path = os.path.join(dir, 'gt_skel_gbl_ori.txt')
        calib_imu_bone_path = os.path.join(dir, 's{}_{}{}_calib_imu_bone.txt'.
                                           format(meta_sub, meta_act, meta_subact))
        calib_imu_ref_path = os.path.join(dir, 's{}_{}{}_calib_imu_ref.txt'.
                                          format(meta_sub, meta_act, meta_subact))
        imu_data_path = os.path.join(dir, 's{}_{}{}_Xsens.sensors'.
                                          format(meta_sub, meta_act, meta_subact))
        bvh_path = os.path.join(dir, '{}{}_BlenderZXY_YmZ.bvh'.
                                          format(meta_act, meta_subact))
        gt_pos = dt.parse_vicon_gt_pos(gt_pos_path)
        gt_ori = dt.parse_vicon_gt_ori(gt_ori_path)
        imu_data = dt.parse_sensor_6axis(imu_data_path)
        calib_imu_bone = dt.parse_calib_imu_bone(calib_imu_bone_path)
        calib_imu_ref = dt.parse_calib_imu_ref(calib_imu_ref_path)
        bone_info = dt.parse_imu_bone_info(bvh_path)
        canvas_size = (1079., 1919.)  # height width

        filtered_joints = config['joints_filter']

        # bone vector / orientation, not camera related
        bones = ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm',
                 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg']
        # obtain ref for all bones
        bone_refs = dict()
        for bone in bones:
            joint_p = bone_info[bone][0]
            joint_c = bone_info[bone][1]
            bone_vec = np.array(bone_info[bone][2]) * 25.4
            q_TI = calib_imu_ref[bone]
            q_bi = calib_imu_bone[bone]
            q_TI = Quaternion(q_TI)
            q_bi = Quaternion(q_bi)
            q_ib = q_bi.conjugate
            bone_refs[bone] = {'joint_p': joint_p, 'joint_c': joint_c,
                               'bone_vec': bone_vec,
                               'q_TI': q_TI, 'q_ib': q_ib}

        bone_vectors = dict()  # of all frames

        for c in range(8):
            mp4_file_name = 'TC_S{}_{}{}_cam{}.mp4'.format(meta_sub, meta_act, meta_subact, c+1)
            mp4_file_path = os.path.join(dir, mp4_file_name)
            cam = cameras[c]
            vid_info = ffprobe(mp4_file_path)
            vid_frame_num = int(vid_info['video']['@nb_frames'])

            # print(mp4_file_name, vid_info['video']['@nb_frames'], len(gt_pos)- int(vid_info['video']['@nb_frames']),
            #       vid_info['video']['@bit_rate'])

            out_path = os.path.join(config['db_out_dir'], 'marked')
            out_path = os.path.join(out_path, 'sub{}_{}_{}_cam{}'.format(meta_sub, meta_act, meta_subact, c+1))
            if config['save_visualization']:
                if not os.path.exists(out_path):
                    os.makedirs(out_path)

            # where to save extract frames
            seq_dir_name = 's_{:0>2}_act_{:0>2}_subact_{:0>2}_ca_{:0>2}'.format(meta_sub, meta['action'], meta_subact, c + 1)
            seq_dir_path = os.path.join(config['db_out_dir'], seq_dir_name)
            if config['save_frame']:
                if not os.path.exists(seq_dir_path):
                    os.makedirs(seq_dir_path)

            vid_ff = vreader(mp4_file_path)
            min_frame_to_iter = min(vid_frame_num, len(gt_pos), len(gt_ori), len(imu_data))
            for idx in tqdm(range(min_frame_to_iter)):
                pose3d = np.zeros([3, len(all_joints)])
                for idx_j, j in enumerate(all_joints):
                    pose3d[:, idx_j] = gt_pos[idx][j]
                pose3d = pose3d * 0.0254  # inch to meter
                pose2d = project_pose3d_to_2d(pose3d, cam, do_distor_corr=True)

                if config['save_visualization'] or config['save_frame']:
                    aframe = next(vid_ff)
                if config['save_visualization']:  # skeleton visualization save to disk
                    out_file_path = os.path.join(out_path, '{:0>6d}.jpg'.format(idx))
                    marked_img = _visualize_one_frame(aframe, pose2d)  # todo vis box on image
                    img_4save = cv2.cvtColor(marked_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_file_path, img_4save)

                # cropping box information
                p2d, p3d_cam, p3d, vis = filter_and_project_2d_pose(gt_pos[idx], filtered_joints,
                                                               cam, canvas_size,
                                                               do_distor_corr=True)
                mvpose_vis = np.reshape([vis/2., vis/2., vis/2.], (3, -1))
                # vis follow coco protocol, divide 2 and copy 3 times to follow mvpose
                root_joint = project_pose3d_to_cam(np.reshape(gt_pos[idx]['Hips'], (3,-1))*0.0254, cam)
                tl_joint = np.copy(root_joint)  # shape (3,1)
                br_joint = np.copy(root_joint)
                tl_joint[0,0] -= 1.0000
                tl_joint[1,0] -= 0.9000
                br_joint[0,0] += 1.0000
                br_joint[1,0] += 1.1000
                bbox_25d = np.concatenate((root_joint, tl_joint, br_joint), axis=1)
                bbox = project_cam_to_uv(bbox_25d, cam, do_distor_corr=True)  # contain 3 point: center, tl, br

                box_center = tuple(bbox[:,0])  # (x, y)
                box_scale = tuple((bbox[:,2] - bbox[:,1])/200.)
                box = tuple(np.concatenate([bbox[:,2], bbox[:,1]]))  # (x_tl, y_tl, x_br, y_br)

                frame_file_name = '{:0>6d}.jpg'.format(idx)
                frame_file_path = os.path.join(seq_dir_path, frame_file_name)
                if config['save_frame']:  # save video frame to disk
                    frame_to_cv = cv2.cvtColor(aframe, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(frame_file_path, frame_to_cv)

                # notice: Difference between totalcapture and h36m project,
                # (1) joints_3d in mm
                # (2) camera['T'] in mm
                # (3) in totalcapture: point_Camera = R.dot(point_Tracking) + T (point and T in m);
                #     in h36m: point_Camera = R.dot(point_Tracking - T)  (point and T in mm)
                #     aka in h36m: point_Tracking = R^{-1}.dot(point_Camera) + T
                # (4) coordinates shape is (num_cords, 3), aka row vector, but I like col vector more
                cam_in_h36m_format = copy.deepcopy(cam)
                cam_in_h36m_format['R'] = cam_in_h36m_format['R']
                cam_in_h36m_format['T'] = cam_in_h36m_format['T']*1000.*(-1.)
                cam_in_h36m_format['T'] = cam_in_h36m_format['R'].T.dot(cam_in_h36m_format['T'])
                del cam_in_h36m_format['intri_mat']
                del cam_in_h36m_format['extri_mat']

                # bone vector
                # avoid parsing in each view, only in first view
                if idx not in bone_vectors:
                    bone_vector_of_one_frame = dict()
                    for bone in bones:
                        q_TI = bone_refs[bone]['q_TI']
                        q_ib = bone_refs[bone]['q_ib']
                        bone_vec = bone_refs[bone]['bone_vec']

                        ori = imu_data[idx][bone][0]
                        q_Ii = Quaternion(ori)
                        q_Tb = q_TI * q_Ii * q_ib
                        rotated_bone_vec = q_Tb.rotate(bone_vec)
                        bone_vector_of_one_frame[bone] = rotated_bone_vec
                    bone_vectors[idx] = bone_vector_of_one_frame

                dataitem = {
                    'image': os.path.join(seq_dir_name, '{:0>6d}.jpg'.format(idx)),
                    'joints_2d': p2d.T,
                    'joints_3d': (p3d_cam*1000.).T,  # 3d pose in camera frame, for psm evaluation
                    'joints_vis': mvpose_vis.T,  # 0: in-visible, 1: visible.
                    'center': box_center,
                    'scale': box_scale,
                    'box': box,
                    'video_id': mp4_file_name,  # mp4 file name  # todo
                    'image_id': idx,
                    'subject': meta['subject'],
                    'action': meta['action'],
                    'subaction': meta['subaction'],
                    'camera_id': c,  # start from 0
                    'camera': cam_in_h36m_format,
                    'source': 'totalcapture',
                    'bone_vec': bone_vectors[idx],
                    'joints_gt': p3d.T*1000.  # groundtruth in tracking frame
                }

                dataset.append(dataitem)

    return dataset


def filter_and_project_2d_pose(p3d, filtered_joints, camera, canvas_size, do_distor_corr=True):
    """

    :param p3d: groundtruth in !inch!, dict: {'joint name': [x,y,z], ... }
    :param filtered_joints: list of joint name strings. joints orders matters
    :param camera: (height, width) aka (row, col)
    :param canvas_size:
    :param do_distor_corr:
    :return:
    """
    pose3d = []
    for j in filtered_joints:
        pose3d.append(p3d[j])
    pose3d = np.array(pose3d).T  # shape(3, nJoints)
    pose3d = pose3d * 0.0254  # inch to meter
    # pose2d = project_pose3d_to_2d(pose3d, camera, do_distor_corr)
    pose3d_cam_frame = project_pose3d_to_cam(pose3d, camera)
    pose2d = project_cam_to_uv(pose3d_cam_frame, camera, do_distor_corr=do_distor_corr)

    # y is 2nd cord of pose2d, and height, row of canvas
    is_y_in_canvas = np.logical_and(pose2d[1, :] >= 0, pose2d[1, :] <= canvas_size[0])
    is_x_in_canvas = np.logical_and(pose2d[0, :] >= 0, pose2d[0, :] <= canvas_size[1])
    is_pt_in_canvas = np.logical_and(is_x_in_canvas, is_y_in_canvas)
    vis = is_pt_in_canvas * 2  # follow coco vis definition, 0: not in image; 1: not visible; 2: fully visible

    return pose2d, pose3d_cam_frame, pose3d, vis


def _visualize_one_frame(frame, pose2d, do_show=False):
    # vicon_joints = ('Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder',
    #                 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm',
    #                 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot')
    filtered_joints = [0, 5,6, 8,9,10, 12,13,14, 15,16,17, 18,19,20]
    uncoonected_joints = [1,2,3,4, 7, 11,]
    connections = [(0,5), (0,18), (0,15), (5,6), (5,8), (8,9), (9,10),
                   (5,12),(12,13),(13,14), (18,19), (19,20), (15,16), (16,17)]
    cc = (255,0,0)
    cr = (0,255,0)
    cl = (0,0,255)
    conn_colors = [cc, cl, cr, cc, cr, cr, cr,
                   cl, cl, cl, cl, cl, cr, cr]

    marked_img = np.copy(frame)
    pose = np.array(pose2d, dtype=np.int).T
    pose = pose.tolist()
    # for cord in pose:
    for fj in filtered_joints:
        cv2.circle(marked_img, tuple(pose[fj]), 5, (255,0,0), -1, lineType=cv2.LINE_AA)

    for fj in uncoonected_joints:
        cv2.circle(marked_img, tuple(pose[fj]), 3, (0,150,255), -1, lineType=cv2.LINE_AA)

    for (p1, p2), color in zip(connections, conn_colors):
        cv2.line(marked_img, tuple(pose[p1]), tuple(pose[p2]), color=color, thickness=2, lineType=cv2.LINE_AA)

    if do_show:
        ax = plt.subplot()
        ax.imshow(marked_img)
        plt.show()
        plt.pause(0)  # wait for close

    return marked_img


def project_pose3d_to_2d(pose3d, camera, do_distor_corr=True):
    """
    refer to https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    :param pose3d: each column is a 3d point
    :param camera:
    :return:
    """
    pose3d = np.array(pose3d)
    num_pts = pose3d.shape[1]
    pts = np.ones([4, num_pts])
    pts[:3, :] = pose3d
    xyz = np.dot(camera['extri_mat'], pts)
    xy1 = np.divide(xyz, xyz[2, :])
    if do_distor_corr:
        r_square = np.power(xy1[0], 2) + np.power(xy1[1], 2)
        radial_distor_coeff = 1. + camera['distor']*r_square  # todo not sure
        xy1 = np.multiply(xy1, np.array([radial_distor_coeff, radial_distor_coeff, np.ones_like(radial_distor_coeff)]))

    uv1 = np.dot(camera['intri_mat'], xy1)
    return uv1[0:2, :]


def project_pose3d_to_cam(pose3d, camera):
    """
    (x,y,z) in tracking frame --> (x',y',z') in camera frame
    :param pose3d:
    :param camera:
    :return:
    """
    pose3d = np.array(pose3d)
    num_pts = pose3d.shape[1]
    pts = np.ones([4, num_pts])
    pts[:3, :] = pose3d
    xyz = np.dot(camera['extri_mat'], pts)
    return xyz


def project_cam_to_uv(xyz, camera, do_distor_corr=True):
    xy1 = np.divide(xyz, xyz[2, :])
    if do_distor_corr:
        r_square = np.power(xy1[0], 2) + np.power(xy1[1], 2)
        radial_distor_coeff = 1. + camera['distor'] * r_square  # todo not sure
        xy1 = np.multiply(xy1, np.array([radial_distor_coeff, radial_distor_coeff, np.ones_like(radial_distor_coeff)]))

    uv1 = np.dot(camera['intri_mat'], xy1)
    return uv1[0:2, :]


if __name__ == '__main__':
    config_file = 'gendata/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)

    # train val dirs [(seq_dir, meta), ...]
    train_dir_meta, test_dir_meta = get_train_val_dirs(config)
    cameras = load_camera(config)

    if config['gen_train']:
        print('-------------------generate {}-----------------------'.format('train'))
        train_dataset = extract_db(config, train_dir_meta, cameras)

        with open(os.path.join(config['db_out_dir'], 'totalcapture_train.pkl'), 'wb') as f:
            pickle.dump(train_dataset, f)

    if config['gen_test']:
        print('-------------------generate {}-----------------------'.format('val/test'))
        test_dataset = extract_db(config, test_dir_meta, cameras)

        with open(os.path.join(config['db_out_dir'], 'totalcapture_validation.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)

    pass
