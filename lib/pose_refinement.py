from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .loss import (pose_mask_loss, regularization_loss, pose_prior_loss, joints_2d_loss)
import sys
import os

import torch
import torch.nn as nn

from .pose_util import smpl_to_pose
sys.path.append(os.path.join('..'))

smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()
    
body25_2_coco = [0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18]
class PoseRefinement(nn.Module):
    """ Implementation of Pose Refinement for a single frame """

    def __init__(self, render=None, camera=None, de_graph=None, 
                 regularization_weight=2e-3, pose_prior_weight=9e0, mask_weight=1.5,
                 optimization_type = 'fixed',
                 step_size=2e-3, batch_size=1, num_iters=500, 
                 max_iter=20, img_size=None, start_frame = 0):
        super().__init__()
        self.render = render
        self.camera = camera
        self.de_graph = de_graph

        self.regularization_weight = regularization_weight
        self.pose_prior_weight = pose_prior_weight
        self.mask_weight = mask_weight

        self.optimization_type = optimization_type

        self.step_size = step_size
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.max_iter = max_iter

        self.img_size = img_size
        self.start_frame = start_frame


    def forward(self, init_betas, init_trans, init_displacement, gt_joints_2d,
                gt_thetas, gt_silhouettes, init_joints_3d, joint_confidence,
                init_d_rotations, init_d_translations, frame):
        """

        """
        
        # Make the these parameters learnable
        opt_betas = init_betas.detach().clone().cuda()
        opt_thetas = gt_thetas.detach().clone().cuda()
        opt_trans = init_trans.detach().clone().cuda()

        # Optimize the SMPL model
        opt_betas.requires_grad = True
        opt_thetas.requires_grad = True
        opt_trans.requires_grad = True

        d_rotations = init_d_rotations.detach()
        d_translations = init_d_translations.detach()
        init_joints_3d = init_joints_3d.detach()

        last_loss = torch.tensor([0]).cuda()
        early_stopping_step = 0
        if frame == self.start_frame:
            self.num_iters = 70
            pose_opt_params = [{'params': opt_betas, 'lr':self.step_size},
                               {'params': opt_thetas, 'lr':self.step_size},
                               {'params': opt_trans, 'lr':self.step_size}] 

            pose_optimizer = torch.optim.Adam(pose_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        else:
            self.num_iters = 40   

            opt_betas.requires_grad = False
            pose_opt_params = [
                              {'params': opt_thetas, 'lr':self.step_size / 2.},
                              {'params': opt_trans, 'lr':self.step_size}] 

            pose_optimizer = torch.optim.Adam(pose_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        for i in range(self.num_iters):
            smpl_output = self.render.get_smpl_output(betas=opt_betas,thetas=opt_thetas, trans=opt_trans, 
                                                      displacement=init_displacement.expand(self.batch_size,-1,-1),
                                                      absolute_displacement=False)

            warpped_vertices, _ = self.de_graph(smpl_output.vertices.squeeze(), d_rotations, d_translations)
            silhouettes, _ = self.render.forward(warpped_vertices, mode='a')  

            if self.img_size[0] < self.img_size[1]:
                silhouettes = silhouettes[:, abs(self.img_size[0]-self.img_size[1])//2:(self.img_size[0]+self.img_size[1])//2, :]
            elif self.img_size[0] > self.img_size[1]:
                silhouettes = silhouettes[:, :, abs(self.img_size[0]-self.img_size[1])//2:(self.img_size[0]+self.img_size[1])//2]
            else:
                silhouettes = silhouettes               
            
            joints_2d = self.camera(torch.index_select(smpl_output.joints, 1, smpl2op_mapping))

            mask_loss = pose_mask_loss(gt_silhouettes, silhouettes)

            reg_loss = regularization_loss(init_joints_3d, smpl_output.joints) # [:,:24,...]

            pprior_loss = pose_prior_loss(opt_thetas[:,3:], opt_betas)

            j2d_loss = joints_2d_loss(gt_joints_2d, joints_2d, joint_confidence)
            
            loss = j2d_loss + self.mask_weight*mask_loss + \
                   self.regularization_weight * reg_loss + self.pose_prior_weight * pprior_loss 

            pose_optimizer.zero_grad()

            if self.optimization_type == 'converge':
                loss.backward()
                pose_optimizer.step()

                if i == 0:
                    last_loss = loss.clone()
                elif early_stopping_step >= 5:
                    print('total iter:', i)
                    break
                elif last_loss.item() < loss.item():
                    early_stopping_step += 1
                    last_loss = loss.clone()
                else:
                    early_stopping_step = 0
                    last_loss = loss.clone()    
            else:
                loss.backward()
                pose_optimizer.step()

        smpl_output = self.render.get_smpl_output(betas=opt_betas,thetas=opt_thetas, trans=opt_trans,
                                                  displacement=init_displacement.expand(self.batch_size,-1,-1),
                                                  absolute_displacement=False)

        opt_joints_3d = smpl_output.joints 
        return opt_betas, opt_thetas, opt_trans, opt_joints_3d
