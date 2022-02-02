from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import argparse
import time
import pickle as pkl
from pyhocon import ConfigFactory
import torch

import numpy as np

import trimesh
from psbody.mesh import Mesh

from mesh_intersection.bvh_search_tree import BVH

from lib.th_SMPL import th_SMPL
smpl_right_hand_idx = np.load('./assets/smpl_right_hand_idx.npy')
smpl_left_hand_idx = np.load('./assets/smpl_left_hand_idx.npy')
binary_mask = np.ones((6890, 3))
binary_mask[smpl_right_hand_idx,:] = 0.
binary_mask[smpl_left_hand_idx,:] = 0.
smpl_hand_faces = np.load('./assets/smpl_hand_faces.npy')
max_collisions = 8

def surface_smoothing(out_dir, gender):
    device = torch.device('cuda')
    model_path = out_dir + '/full_smpld.pkl'
    model = pkl.load(open(model_path, 'rb'), encoding='latin1')
    
    num_iter = 0
    while True:
        if num_iter == 0:
            mesh_fn = out_dir + '/full_smpld.obj'
            ps_mesh = Mesh(filename=mesh_fn)
            input_mesh = trimesh.Trimesh(vertices=ps_mesh.v, faces=ps_mesh.f, process=False)
        else:    
            mesh_fn = out_dir + '/full_smpld_sm.obj'
            ps_mesh = Mesh(filename=mesh_fn)
            input_mesh = trimesh.load(mesh_fn, process=False)
        
        vertices = torch.tensor(input_mesh.vertices,
                                dtype=torch.float32, device=device)
        faces = torch.tensor(input_mesh.faces.astype(np.int64),
                            dtype=torch.long,
                            device=device)

        
        triangles = vertices[faces].unsqueeze(dim=0)

        torch.cuda.synchronize()
        start = time.time()
        m = BVH(max_collisions=max_collisions)
        outputs = m(triangles)
        torch.cuda.synchronize()
        print('Elapsed time', time.time() - start)

        outputs = outputs.detach().cpu().numpy().squeeze()

        collisions = outputs[outputs[:, 0] >= 0, :]

        excl_collisions = []
        # hand faces excluded
        for c_fs in collisions:
            if c_fs[0] not in smpl_hand_faces and c_fs[1] not in smpl_hand_faces:
                excl_collisions.append(c_fs)
        excl_collisions = np.array(excl_collisions)

        print(excl_collisions.shape)

        print('Number of collisions = ', excl_collisions.shape[0])
        if excl_collisions.shape[0] == 0:
            break
        print('Percentage of collisions (%)',
            excl_collisions.shape[0] / float(triangles.shape[1]) * 100)        

        recv_faces = input_mesh.faces[excl_collisions[:, 0]]
        intr_faces = input_mesh.faces[excl_collisions[:, 1]]

        collisions_faces = np.concatenate((recv_faces, intr_faces))

        uniques, _ = np.unique(collisions_faces, axis=0, return_index=True)
        sorted_uniques = np.unique(uniques)
        values = np.arange(len(sorted_uniques))  
        norm_uniques = np.zeros_like(uniques)

        table = {}

        for u in range(len(sorted_uniques)):
            table[sorted_uniques[u]] = values[u]
        for i in range(uniques.shape[0]):
            for j in range(uniques.shape[1]):
                 norm_uniques[i,j] = table[uniques[i,j]]

        referenced = np.zeros(len(input_mesh.vertices), dtype=bool)

        referenced[uniques] = True
        mask = np.where(referenced==True)[0]

        collisions_mesh = trimesh.Trimesh(input_mesh.vertices[sorted_uniques].copy(), norm_uniques, process=False)

        trimesh.smoothing.filter_humphrey(collisions_mesh, alpha=0.1, beta=0.5, iterations=20)

        input_mesh.vertices[mask] = collisions_mesh.vertices

        trimesh.exchange.export.export_mesh(input_mesh, out_dir + '/full_smpld_sm.obj')

        num_iter += 1

    mesh_fn = out_dir + '/full_smpld_sm.obj'
    final_mesh = trimesh.load(mesh_fn, process=False)

    smpl = th_SMPL(betas = torch.tensor(model['betas']), trans=torch.tensor(model['trans']), pose = torch.tensor(model['pose']), gender=gender)
    vertices, th_T = smpl.forward(return_lbs = True)

    temp_verts = torch.cat([vertices.type(th_T.dtype), torch.ones((final_mesh.vertices.shape[0], 1), dtype=th_T.dtype, device=th_T.device),], 1).unsqueeze(-1)
    temp_verts_T = torch.matmul(th_T.transpose(0,3)[..., 0].inverse(), temp_verts)
    
    homo_verts = torch.cat([torch.tensor(final_mesh.vertices).type(th_T.dtype), torch.ones((final_mesh.vertices.shape[0], 1), dtype=th_T.dtype, device=th_T.device),], 1).unsqueeze(-1)
    homo_verts_T = torch.matmul(th_T.transpose(0,3)[..., 0].inverse(), homo_verts)
    offsets_sm_T = homo_verts_T[:, :3, 0].detach().cpu().numpy() - temp_verts_T[:, :3, 0].detach().cpu().numpy()

    offsets_sm_T *= binary_mask
    smpl_sm_dict = {'pose': model['pose'], 'betas': model['betas'], 'trans': model['trans'], 'offsets': offsets_sm_T}

    pkl.dump(smpl_sm_dict, open(out_dir+'/full_smpld_sm.pkl', 'wb'))
    # output_model_path = out_dir + 'full_smpld_sm.pkl'
    # test_model = pkl.load(open(output_model_path, 'rb'), encoding='latin1')

    smpl_w_offsets = th_SMPL(betas = torch.tensor(smpl_sm_dict['betas']), trans = torch.tensor(smpl_sm_dict['trans']), offsets = torch.tensor(smpl_sm_dict['offsets']), gender=gender)
    # T-pose vertices
    vertices = smpl_w_offsets.forward()
    T_mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), input_mesh.faces, process=False)

    trimesh.exchange.export.export_mesh(T_mesh, out_dir + '/full_smpld_sm_T.obj')
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--out_dir', type=str, help='path to the registered mesh', default='test_output')
    args = parser.parse_args()
    
    out_dir = args.out_dir
    conf = ConfigFactory.parse_file(args.config)
    gender = conf.get_string('gender')
    surface_smoothing(out_dir, gender)