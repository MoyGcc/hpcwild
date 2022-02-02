"""
Scans need to be processed before passing them to IPNet.
Users would have to modify the script a bit to suit their I/O. function func can be directly used.
https://github.com/bharat-b7/IPNet
"""

import os
from os.path import join, split, exists
from shutil import copyfile
from glob import glob
from psbody.mesh import Mesh
import trimesh
import numpy as np
from pyhocon import ConfigFactory
bb_min = -1.
bb_max = 1.

new_cent = (bb_max + bb_min) / 2
SCALE = 1.5
MIN_SCALE = 1.6


def func(vertices, scale=None, cent=None):
    """
    Function to normalize the scans for IPNet.
    Ensure that the registration and body are normalized in the same way as scan.
    """
    # import pdb
    # pdb.set_trace()
    if scale is None:
        scale = max(MIN_SCALE, vertices[:, 1].max() - vertices[:, 1].min())
    
    vertices /= (scale / SCALE)

    if cent is None:
        cent = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices -= (cent - new_cent)

    return vertices, scale, cent


def process(config, path):
    conf = ConfigFactory.parse_file(config)
    scan = 'results/pifuhd_final/recon'
    name = 'result_%s_%s_%d_256' %  (conf.get_string('dataset'), conf.get_string('seq'), conf.get_int('frame_num'))
    # Load scan and normalize
    
    mesh = Mesh(filename=join(scan, name + '.obj')) 
    mesh.v, scale, cent = func(mesh.v)
    # mesh.vt = mesh.vt[:, :2]
    mesh.write_obj(join(path, name + '_scaled.obj'))
    # np.save(join(path, name + '_cent.npy'), [scale / SCALE, (cent - new_cent)])

    # Normalize the registration according to scan.
    if exists(join(scan, name + '_reg.obj')):
        reg = Mesh(filename=join(scan, name + '_reg.obj'))  
        reg.v, _, _ = func(reg.v, scale, cent)
        reg.write_obj(join(path, name + '_scaled_reg.obj'))
        count_reg = 1
    else:
        count_reg = 0

    # normalize the body under clothing.
    if exists(join(scan, name + '_naked.obj')):
        mesh = Mesh(filename=join(scan, name + '_naked.obj'))  
        mesh.v, _, _ = func(mesh.v, scale, cent)
        mesh.write_obj(join(path, name + '_scaled_naked.obj'))
        count_body = 1
    else:
        count_body = 0

    print('Done,', name)
    return count_reg, count_body


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--path', type=str, default='test_output')
    args = parser.parse_args()
    
    _, _ = process(args.config, args.path)
