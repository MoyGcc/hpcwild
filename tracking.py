from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import argparse
from pyhocon import ConfigFactory
import pickle as pkl
from tqdm import trange
import numpy as np
import cv2

import torch

from lib.pose_refinement import PoseRefinement
from lib.surface_refinement import SurfaceRefinement
from lib.smoothing import smoothing
from smpl_rendering.smpl_renderer import SMPLRenderer

from lib.camera import PerspectiveCamera
from lib.utils import rectify_pose
from lib.deformation_graph import DeformationGraph
from psbody.mesh import Mesh

hip_index = 8

def fit(dataset_path, seq_name, model_dir, actor):

    batch_size = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    seq_name = seq_name
    datasetDir = dataset_path
    root = os.getcwd()
    obj_path = os.path.join(root, '%s/full_smpld_sm_T.obj' % model_dir)
    model_path = os.path.join(root, './%s/full_smpld_sm.pkl' % model_dir) 
    output_dir = os.path.join(root, 'test_data/result', seq_name+'_'+str(actor))
    output_img_dir = os.path.join(output_dir, 'front_no_smoothing')

    obj_T = Mesh(filename=obj_path)
    full_model = pkl.load(open(model_path, 'rb'), encoding='latin1')

    seq_path = os.path.join(datasetDir, 'sequenceFiles/test', seq_name+'.pkl')

    thetas_file = pkl.load(open(os.path.join(root, 'test_data/%s.pkl' % seq_name), 'rb'), encoding='latin1')['thetas'][actor]
    op_outputs = np.load('./test_data/openpose_jnts.npy')
    start_frame = 0
    
    seq = pkl.load(open(seq_path, 'rb'), encoding='latin1')

    gender = seq['genders'][actor]

    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    
    input_img_path = os.path.join(datasetDir,'imageFiles',seq_name+'/image_{:05d}.jpg'.format(start_frame))

    input_img = cv2.imread(input_img_path)

    # resize to the half of the original image
    resize_factor = 2
    input_img = cv2.resize(input_img, (int(input_img.shape[1]/resize_factor), int(input_img.shape[0]/resize_factor)))
    render_image_size = max(input_img.shape[0], input_img.shape[1])

    cam_intrinsics = seq['cam_intrinsics']/resize_factor # resize
    half_max_length = max(cam_intrinsics[0:2,2]) 
    principal_point = [-(cam_intrinsics[0,2]-input_img.shape[1]/2.)/(input_img.shape[1]/2.), -(cam_intrinsics[1,2]-input_img.shape[0]/2.)/(input_img.shape[0]/2.)] # - or +
    principal_point = torch.tensor(principal_point).unsqueeze(0)

    f = torch.tensor([(cam_intrinsics[0,0]/half_max_length).astype(np.float32), (cam_intrinsics[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)

    center = torch.tensor(cam_intrinsics[0:2,2]).unsqueeze(0)

    renderer = SMPLRenderer(batch_size, render_image_size, f, principal_point, gender=gender).cuda()   
    
    opt_betas = full_model['betas'][:10]
    opt_betas = torch.tensor(opt_betas.astype(np.float32), requires_grad=False).unsqueeze(0).cuda().expand(batch_size, -1)
    init_displacement = torch.from_numpy(full_model['offsets'].astype(np.float32)).cuda().float()
 
    # deformation graph
    de_graph = DeformationGraph() 
    de_graph.construct_graph(obj_T.v.astype(np.float32), renderer.faces.cpu().numpy().squeeze())
    num_nodes = de_graph.nodes_idx.shape[0]
    
    # guess initial trans 
    mask = cv2.imread('./test_data/mask/mask_%04d.png' % (start_frame))[..., 0]
    y = np.where(mask != 0)[0]
    gt_joints_2d = op_outputs[start_frame]
    h2d = y.max() - y.min()

    if gender == 'female':
        guess_height = 1.6
    if gender == 'male':
        guess_height = 1.75

    T_hip = renderer.get_T_hip(opt_betas).data.cpu().numpy().reshape(3,1)
    
    init_trans_z = np.array(cam_intrinsics[0,0] * guess_height / h2d)
    init_hip_2d = gt_joints_2d[hip_index]
    init_trans_x = ((init_hip_2d[0] - cam_intrinsics[0,2]) * init_trans_z / cam_intrinsics[0,0]) - T_hip[0]
    init_trans_y = ((init_hip_2d[1] - cam_intrinsics[1,2]) * init_trans_z / cam_intrinsics[1,1]) - T_hip[1]

    opt_d_rotations = torch.zeros((batch_size, num_nodes, 3), requires_grad=True).cuda() # axis angle 
    opt_d_translations = torch.zeros((batch_size, num_nodes, 3), requires_grad=True).cuda()

    T_smpl = np.array([init_trans_x[0], init_trans_y[0], float(init_trans_z)]).reshape(3,1) 
    
    cam = PerspectiveCamera(focal_length_x = torch.tensor((cam_intrinsics[0,0].astype(np.float32))), 
                            focal_length_y = torch.tensor(cam_intrinsics[1,1].astype(np.float32)), 
                            center=center).cuda()

    opt_trans = T_smpl.reshape(3)
    opt_trans = torch.tensor(opt_trans.astype(np.float32)).unsqueeze(0).cuda().expand(batch_size, -1)

    opt_thetas = thetas_file[start_frame].copy()
    
    opt_thetas = rectify_pose(opt_thetas)
    opt_thetas = torch.tensor(opt_thetas.astype(np.float32)).unsqueeze(0).cuda().expand(batch_size, -1)

    smpl_output = renderer.get_smpl_output(betas=opt_betas,thetas=opt_thetas, trans=opt_trans, 
                                         displacement=init_displacement.expand(batch_size,-1,-1),
                                         absolute_displacement=False)
    opt_joints_3d = smpl_output.joints
    
    pr = PoseRefinement(render=renderer, camera=cam, de_graph=de_graph, 
                        img_size=input_img.shape, start_frame=start_frame).cuda()

    sr = SurfaceRefinement(render=renderer, de_graph=de_graph, 
                           img_size=input_img.shape, start_frame=start_frame).cuda()
 
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    output_trans = []
    output_verts = []
    print("start tracking")

    for frame in trange(start_frame, seq['cam_poses'].shape[0]):
        # print("current frame:", frame)
        input_img_path = os.path.join(datasetDir,'imageFiles', seq_name+'/image_{:05d}.jpg'.format(frame))
        input_img = cv2.imread(input_img_path)
        input_img = cv2.resize(input_img, (int(input_img.shape[1]/resize_factor), int(input_img.shape[0]/resize_factor)))

        gt_silhouettes = cv2.imread('./test_data/mask/mask_%04d.png' % frame)[:, :, 0]
        gt_silhouettes = torch.tensor(gt_silhouettes.astype(np.float32)/255.0).unsqueeze(0).cuda()

        op_output = op_outputs[frame]
        
        gt_joints_2d = torch.from_numpy(op_output[:, :2].astype(np.int32)).unsqueeze(0).cuda()

        gt_thetas = torch.tensor(rectify_pose(thetas_file[frame].astype(np.float32))).unsqueeze(0).cuda().expand(batch_size, -1)

        joint_confidence = torch.from_numpy(op_output[:, 2].astype(np.float32)).reshape((-1,1)).cuda()

        opt_betas, opt_thetas, opt_trans, opt_joints_3d = pr.forward(opt_betas, opt_trans, 
                                                                     init_displacement, gt_joints_2d,
                                                                     gt_thetas, gt_silhouettes, opt_joints_3d, joint_confidence, 
                                                                     opt_d_rotations, opt_d_translations, frame)

        warpped_vertices, opt_betas, opt_d_rotations, opt_d_translations = sr.forward(opt_betas, opt_thetas, 
                                                                                      opt_trans, opt_d_rotations, 
                                                                                      opt_d_translations, init_displacement,
                                                                                      gt_silhouettes, frame)

        output_verts.append(warpped_vertices.detach().cpu().numpy().squeeze())
        output_trans.append(opt_trans.detach().cpu().numpy().squeeze())

        # rendered_img, _ = renderer.forward(warpped_vertices.detach(), mode='vis')  
        # if input_img.shape[0] < input_img.shape[1]:
        #     rendered_img = rendered_img[0, abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...].cpu().numpy() * 255   
        # else:
        #     rendered_img = rendered_img[0, :,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2].cpu().numpy() * 255     
        # valid_mask = (rendered_img[:,:,-1] > 0)[:, :, np.newaxis]
        # output_img = (rendered_img[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)

        # cv2.imwrite(output_img_dir + '/{:04d}.png'.format(frame),output_img)

    output_dict = {'verts': output_verts, 'trans': output_trans}
    pkl.dump(output_dict, open(os.path.join(output_dir, 'output_nons.pkl'), 'wb'))


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    conf = args.config
    conf = ConfigFactory.parse_file(conf)
    dataset_path = conf.get_string('dataset_path') 
    seq = conf.get_string('seq')

    actor = 0 # only one actor in this sequence
    model_dir = 'test_output'
    fit(dataset_path=dataset_path, seq_name=seq, model_dir=model_dir, actor=actor)
    smoothing(dataset_path=dataset_path, seq_name=seq, actor=actor)
