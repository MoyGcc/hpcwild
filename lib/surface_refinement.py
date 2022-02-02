from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .loss import pose_mask_loss

import torch
import torch.nn as nn

class SurfaceRefinement(nn.Module):
    """ Implementation of Surface Refinement for a single frame"""

    def __init__(self, render=None, de_graph=None, step_size=1e-3, 
                 batch_size=1, optimization_type = 'fixed',
                 num_iters=5, max_iter=20, 
                 arap_weight=3e0, img_size=None, start_frame = 0):
        super().__init__()

        self.render = render
        self.de_graph = de_graph

        self.step_size = step_size
        self.batch_size = batch_size
        self.optimization_type = optimization_type

        self.num_iters = num_iters
        self.max_iter = max_iter

        self.arap_weight = arap_weight
        self.img_size = img_size
        self.start_frame = start_frame
    def forward(self, init_betas=None, thetas=None, 
                trans=None, init_d_rotations=None, init_d_translations=None, init_displacement=None, gt_silhouettes=None, frame=None):
        # Make the these parameters learnable
        opt_betas = init_betas.detach().clone().cuda()
        opt_d_rotations = init_d_rotations.detach().clone().cuda()
        opt_d_translations = init_d_translations.detach().clone().cuda()

        # Optimize only the non-rigid deformation
        opt_betas.requires_grad = False
        opt_d_rotations.requires_grad = True
        opt_d_translations.requires_grad = True

        thetas.requires_grad = False
        trans.requires_grad = False

        surface_opt_params = [opt_d_rotations, opt_d_translations] 
        surface_optimizer = torch.optim.Adam(surface_opt_params, lr=self.step_size, betas=(0.9, 0.999))
        last_loss = torch.tensor([0]).cuda()
        if frame == self.start_frame:
            self.num_iters = 10
        else:
            self.num_iters = 7
        for i in range(self.num_iters):
            smpl_output = self.render.get_smpl_output(betas=opt_betas, thetas=thetas, trans=trans,
                                                      displacement=init_displacement.expand(self.batch_size,-1,-1),
                                                      absolute_displacement=False)
            smpl_vertices = smpl_output.vertices.squeeze()

            warpped_vertices, arap_loss = self.de_graph(smpl_vertices, opt_d_rotations, opt_d_translations)
            silhouettes, _ = self.render.forward(warpped_vertices, mode='a')
            if self.img_size[0] < self.img_size[1]:
                silhouettes = silhouettes[:, abs(self.img_size[0]-self.img_size[1])//2:(self.img_size[0]+self.img_size[1])//2, :]
            elif self.img_size[0] > self.img_size[1]:
                silhouettes = silhouettes[:, :, abs(self.img_size[0]-self.img_size[1])//2:(self.img_size[0]+self.img_size[1])//2]
            else:  
                silhouettes = silhouettes

            mask_loss = pose_mask_loss(gt_silhouettes, silhouettes) 
            loss = mask_loss + self.arap_weight * arap_loss 
            surface_optimizer.zero_grad()

            if self.optimization_type == 'converge':
                loss.backward()
                surface_optimizer.step()
                if i == 0:
                    last_loss = loss.clone()
                elif last_loss.item() - loss.item() > 1e-6:

                    last_loss = loss.clone()
                else:
                    print('total iter:', i)
                    break
            else:
                loss.backward()
                surface_optimizer.step()

        smpl_output = self.render.get_smpl_output(betas=opt_betas, thetas=thetas, trans=trans,
                                                    displacement=init_displacement.expand(self.batch_size,-1,-1),
                                                    absolute_displacement=False)
        smpl_vertices = smpl_output.vertices.squeeze()
        warpped_vertices, _ = self.de_graph(smpl_vertices, opt_d_rotations, opt_d_translations)
        return warpped_vertices, opt_betas, opt_d_rotations, opt_d_translations
