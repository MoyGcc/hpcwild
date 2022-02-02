from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from .prior import MaxMixturePrior
from .utils import GMoF

num_joints = 25
joints_to_ign = [1,9,12]
joint_weights = torch.ones(num_joints)   
joint_weights[joints_to_ign] = 0

joint_weights = joint_weights.reshape((-1,1)).cuda()

body_pose_prior = MaxMixturePrior().cuda()

robustifier = GMoF(rho=100)

def pose_mask_loss(gt_silhouettes, silhouettes):
    mask_loss = torch.mean((silhouettes - gt_silhouettes) ** 2) 
    return mask_loss

def regularization_loss(init_joints_3d, joints_3d):
    reg_loss = torch.mean((init_joints_3d-joints_3d) ** 2)
    return reg_loss

def joints_2d_loss(gt_joints_2d=None, joints_2d=None, joint_confidence=None):
    joint_diff = robustifier(gt_joints_2d - joints_2d)
    joints_2dloss = torch.sum((joint_confidence*joint_weights) ** 2 * joint_diff)

    return joints_2dloss

def pose_prior_loss(body_pose=None, betas=None):

    pprior_loss = torch.sum(body_pose_prior(body_pose, betas))
    return pprior_loss

