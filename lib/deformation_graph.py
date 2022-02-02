# Adapted from https://github.com/nintendops/DynamicFusion_Body/blob/b6b31d890974cc870d9b439fd9647d5934eeedaa/core/fusion_dm.py

import sys
import numpy as np
import torch
import torch.nn as nn

from scipy.spatial import KDTree

from psbody.mesh import Mesh

from .mesh_sampling import generate_transform_matrices

from .utils import col, batch_rodrigues

eps = sys.float_info.epsilon

class DeformationGraph(nn.Module):

    def __init__(self, radius=0.045, k=3, sampling_strategy='qslim'):
        super().__init__()
        
        self.radius = radius
        self.k = k
        self.max_neigh_num = 18
        self.sampling_strategy = sampling_strategy
        self.one_ring_neigh = []
        self.nodes_idx = None
        self.weights = None
        self.influence_nodes_idx = []
        self.dists = []

    def construct_graph(self, vertices=None, faces=None):
        if self.sampling_strategy == 'qslim':
            m = Mesh(v=vertices, f=faces)
            M, A, D = generate_transform_matrices(m, [10])
            nodes_v = M[1].v
            self.nodes_idx = D[0].nonzero()[1]
            adj_mat = A[1].toarray()
            for i in range(adj_mat.shape[0]):
                self.one_ring_neigh.append(adj_mat[i].nonzero()[0].tolist() + [i]*(self.max_neigh_num-len(adj_mat[i].nonzero()[0])))
            self.one_ring_neigh = torch.tensor(self.one_ring_neigh).cuda()  

        # construct kd tree
        kdtree = KDTree(nodes_v)
        
        for vert in vertices:
            dist, idx = kdtree.query(vert, k=self.k)
            self.dists.append(dist)
            self.influence_nodes_idx.append(idx)
            
        self.weights = -np.log(np.array(self.dists)+eps)
        
        # weights normalization
        self.weights = torch.tensor(self.weights/col(self.weights.sum(1))).cuda()

        self.influence_nodes_idx = torch.tensor(self.influence_nodes_idx).cuda()
        
    def forward(self, vertices, opt_d_rotations, opt_d_translations):
        
        opt_d_rotmat = batch_rodrigues(opt_d_rotations[0]).unsqueeze(0) # 1 * N_c * 3 * 3
        nodes = vertices[self.nodes_idx, ...]

        influence_nodes_v = nodes[self.influence_nodes_idx.reshape((-1,))]# .reshape((6890,3,3))
        opt_d_r = opt_d_rotmat[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3,3)) 
        opt_d_t = opt_d_translations[0, self.influence_nodes_idx.reshape((-1,)), ...]# .reshape((6890,3,3))

        warpped_vertices = (torch.einsum('bij, bkj->bki', opt_d_r, (vertices.repeat_interleave(3, dim=0) - influence_nodes_v).unsqueeze(1)).squeeze(1) \
                            + influence_nodes_v + opt_d_t).reshape((6890,3,3)) * (self.weights.unsqueeze(-1))

        warpped_vertices = warpped_vertices.sum(axis=1).float()

        # opt_dnode_t = 
        
        diff_term = (nodes + opt_d_translations[0]).repeat_interleave(self.max_neigh_num, dim=0) - \
                    (nodes[self.one_ring_neigh.reshape((-1,))] + opt_d_translations[0][self.one_ring_neigh.reshape((-1,))]) - \
                     torch.einsum('bij, bkj->bki', opt_d_rotmat[0].repeat_interleave(self.max_neigh_num, dim=0), \
                    (nodes.repeat_interleave(self.max_neigh_num, dim=0) - nodes[self.one_ring_neigh.reshape((-1,))]).unsqueeze(1)).squeeze(1)
        arap_loss = torch.sum(diff_term ** 2) / self.nodes_idx.shape[0]
        
        return warpped_vertices.unsqueeze(0), arap_loss
