'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
https://github.com/bharat-b7/IPNet
'''
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
from lib.smpl_layer import SMPL_Layer
from lib.body_objectives import torch_pose_obj_data
from lib.torch_functions import batch_sparse_dense_matmul
# smpl_head_idx = pkl.load(open('/home/chen/IPNet/assets/smpl_parts_dense.pkl', 'rb'), encoding='latin1')['head']
# smpl_face_idx = np.load('/home/chen/IPNet/assets/smpl_face_idx.npy')
smpl_right_hand_idx = np.load('/home/chen/IPNet/assets/smpl_right_hand_idx.npy')
smpl_left_hand_idx = np.load('/home/chen/IPNet/assets/smpl_left_hand_idx.npy')
binary_mask = torch.ones((1, 6890, 3)).cuda()
# binary_mask[:,smpl_face_idx,:] = 0.
binary_mask[:,smpl_right_hand_idx,:] = 0.
binary_mask[:,smpl_left_hand_idx,:] = 0.

class th_batch_SMPL_split_params(nn.Module):
    """
    Alternate implementation of th_batch_SMPL that allows us to independently optimise:
     1. global_pose
     2. remaining other_pose
     3. top betas (primarly adjusts bone lengths)
     4. other betas
    """
    def __init__(self, batch_sz, top_betas=None, other_betas=None, global_pose=None, other_pose=None, trans=None,
                 offsets=None, faces=None, gender='male'):
        super(th_batch_SMPL_split_params, self).__init__()
        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, 2))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, 298))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        if other_pose is None:
            self.other_pose = nn.Parameter(torch.zeros(batch_sz, 69))
        else:
            assert other_pose.ndim == 2
            self.other_pose = nn.Parameter(other_pose)

        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)

        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.other_pose], axis=1)

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models')

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

    def forward(self):
        self.betas = torch.cat([self.top_betas, self.other_betas], axis=1)
        self.pose = torch.cat([self.global_pose, self.other_pose], axis=1)
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smpl(self.pose,
                                  th_betas=self.betas,
                                  th_trans=self.trans,
                                  th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class th_batch_SMPL(nn.Module):
    def __init__(self, batch_sz, betas=None, pose=None, trans=None, offsets=None, faces=None, gender='male'):
        super(th_batch_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, 72))
        else:
            assert pose.ndim == 2
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890,3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.faces = faces
        self.gender = gender
        # pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                               model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models')

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = torch_pose_obj_data(batch_size=batch_sz)

    def forward(self):
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                              th_betas=self.betas,
                                              th_trans=self.trans,
                                              th_offsets=self.offsets)
        return verts, jtr, tposed, naked
    def get_vertices_clean(self):
        self.offsets_clean = self.offsets.detach().clone()
        self.offsets_clean = self.offsets_clean * binary_mask
        # self.offsets_clean[:,smpl_head_idx,:] /= 2.
        verts, jtr, tposed, naked = self.smpl(self.pose,
                                   th_betas=self.betas,
                                   th_trans=self.trans,
                                   th_offsets=self.offsets_clean)
        return verts, jtr, tposed, naked
    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""

        verts, _, _, _ = self.smpl(self.pose,
                                  th_betas=self.betas,
                                  th_trans=self.trans,
                                  th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands


class th_SMPL(nn.Module):
    def __init__(self, betas=None, pose=None, trans=None, offsets=None, gender='male'):
        super(th_SMPL, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(6890,3))
        else:
            self.offsets = nn.Parameter(offsets)

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender,
                          model_root='/BS/bharat/work/installation/smplpytorch/smplpytorch/native/models')

    def forward(self, return_lbs = False):
        if return_lbs == True:
            verts, Jtr, tposed, naked, th_T = self.smpl(self.pose.unsqueeze(axis=0),
                                                        th_betas=self.betas.unsqueeze(axis=0),
                                                        th_trans=self.trans.unsqueeze(axis=0),
                                                        th_offsets=self.offsets.unsqueeze(axis=0),
                                                        return_lbs = return_lbs)
            return verts[0], th_T
        else:

            verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                                th_betas=self.betas.unsqueeze(axis=0),
                                                th_trans=self.trans.unsqueeze(axis=0),
                                                th_offsets=self.offsets.unsqueeze(axis=0),
                                                return_lbs = return_lbs)
            return verts[0]
