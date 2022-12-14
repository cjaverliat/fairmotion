# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions
from fairmotion.utils import utils
from tqdm import tqdm

"""
Structure of npz file in AMASS dataset is as follows.
- trans (num_frames, 3):  translation (x, y, z) of root joint
- gender str: Gender of actor
- mocap_framerate int: Framerate in Hz
- betas (16): Shape parameters of body. See https://smpl.is.tue.mpg.de/
- dmpls (num_frames, 8): DMPL parameters
- poses (num_frames, 156): Pose data. Each pose is represented as 156-sized
    array. The mapping of indices encoding data is as follows:
    0-2 Root orientation
    3-65 Body joint orientations
    66-155 Finger articulations
"""

# Custom names for 22 joints in AMASS data
body_joint_names = [
    "root",
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
]

hand_joint_names = [
    "rindex0",
    "rindex1",
    "rindex2",
    "rmiddle0",
    "rmiddle1",
    "rmiddle2",
    "rpinky0",
    "rpinky1",
    "rpinky2",
    "rring0",
    "rring1",
    "rring2",
    "rthumb0",
    "rthumb1",
    "rthumb2",
    "lindex0",
    "lindex1",
    "lindex2",
    "lmiddle0",
    "lmiddle1",
    "lmiddle2",
    "lpinky0",
    "lpinky1",
    "lpinky2",
    "lring0",
    "lring1",
    "lring2",
    "lthumb0",
    "lthumb1",
    "lthumb2"
]

def create_skeleton_from_amass_bodymodel(bm, betas, body_num_joints, hand_num_joints, joint_names):
    pose_body_zeros = torch.zeros((1, 3 * (body_num_joints - 1))) # 22 body joints
    pose_hand_zeros = torch.zeros((1, 3 * hand_num_joints)) # 30 hand joints (15 per hand), 
    body = bm(pose_body=pose_body_zeros, pose_hand=pose_hand_zeros, betas=betas)
    base_position = body.Jtr.detach().numpy()[0, 0:body_num_joints+hand_num_joints]
    parents = bm.kintree_table[0].long()[:body_num_joints+hand_num_joints]

    joints = []
    for i in range(body_num_joints + hand_num_joints):
        joint = motion_class.Joint(name=joint_names[i])
        if i == 0: # root joint
            joint.info["dof"] = 6
            joint.xform_from_parent_joint = conversions.p2T(np.zeros(3))
        else:
            joint.info["dof"] = 3
            joint.xform_from_parent_joint = conversions.p2T(
                base_position[i] - base_position[parents[i]]
            )
        joints.append(joint)

    parent_joints = []
    for i in range(body_num_joints + hand_num_joints):
        parent_joint = None if parents[i] < 0 else joints[parents[i]]
        parent_joints.append(parent_joint)

    skel = motion_class.Skeleton()
    for i in range(body_num_joints + hand_num_joints):
        skel.add_joint(joints[i], parent_joints[i])

    return skel


def create_motion_from_amass_data(filename, bm, override_betas=None):
    bdata = np.load(filename)

    if override_betas is not None:
        betas = torch.Tensor(override_betas[:10][np.newaxis]).to("cpu")
    else:
        betas = torch.Tensor(bdata["betas"][:10][np.newaxis]).to("cpu")
    
    skel = create_skeleton_from_amass_bodymodel(
        bm, betas, len(body_joint_names), len(hand_joint_names), body_joint_names + hand_joint_names
    )

    if "mocap_frame_rate" in bdata.files:
        fps = float(bdata["mocap_frame_rate"])
    elif "mocap_framerate" in bdata.files:
        fps = float(bdata["mocap_framerate"])
    else:
        raise Exception(f"Mocap frame rate no found in file {filename}.")

    root_orient = bdata["poses"][:, :3]  # controls the global root orientation
    root_trans = bdata["trans"][:, :3]

    pose_body = bdata["poses"][:, 3:66]  # controls body joint angles
    pose_hand = bdata["poses"][:, 66:] # controls the finger articulation

    motion = motion_class.Motion(skel=skel, fps=fps)

    for frame in range(pose_body.shape[0]):
        root_orient_frame = root_orient[frame]
        root_trans_frame = root_trans[frame]

        pose_body_frame = pose_body[frame]
        pose_hand_frame = pose_hand[frame]

        pose_data = []
        for j in range(len(body_joint_names)):
            if j == 0: # root joint
                T = conversions.Rp2T(
                    conversions.A2R(root_orient_frame), root_trans_frame
                )
            else:
                T = conversions.R2T(
                    conversions.A2R(
                        pose_body_frame[(j - 1) * 3 : (j - 1) * 3 + 3]
                    )
                )
            pose_data.append(T)
    
        for k in range(len(hand_joint_names)):
            T = conversions.R2T(
                conversions.A2R(
                    pose_hand_frame[k * 3 : k * 3 + 3]
                )
            )
            pose_data.append(T)

        motion.add_one_frame(pose_data)

    return motion


def load_body_model(bm_path, num_betas=10):
    comp_device = torch.device("cpu")
    bm = BodyModel(
        bm_fname=bm_path, 
        num_betas=num_betas
    ).to(comp_device)
    return bm


def load(file, bm=None, bm_path=None, num_betas=10, override_betas=None):
    if bm is None:
        # Download the required body model. For SMPL-H download it from
        # http://mano.is.tue.mpg.de/.
        assert bm_path is not None, "Please provide SMPL body model path"
        bm = load_body_model(bm_path, num_betas)
    return create_motion_from_amass_data(
        filename=file, bm=bm, override_betas=override_betas)


def save():
    raise NotImplementedError("Using bvh.save() is recommended")


def load_parallel(files, cpus=20, **kwargs):
    return utils.run_parallel(load, files, num_cpus=cpus, **kwargs)
