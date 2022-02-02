from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import trimesh


if __name__ == "__main__":

    out_dir = 'out_dir71'
    obj = ''
    mesh_fn = './%s/%s.obj' % (out_dir, obj)

    input_mesh = trimesh.load(mesh_fn, process=False)
    
    input_mesh = trimesh.graph.split(input_mesh, only_watertight=False)[0]

    trimesh.exchange.export.export_mesh(input_mesh, './%s/%s_cleaned.obj' % (out_dir, obj))
