from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
from tqdm import trange
import pickle as pkl
import numpy as np
import cv2
import torch

sys.path.append(os.path.join('..'))
from smpl_rendering.smpl_renderer import SMPLRenderer

def smoothing(dataset_path, seq_name, actor):
    batch_size = 1

    seq_name = seq_name
    folder = 'test'
    datasetDir = dataset_path
    root = os.getcwd()
    nons_file = pkl.load(open(os.path.join(root, 'test_data/result', seq_name+'_'+str(actor), 'output_nons.pkl'), 'rb'), encoding='latin1')
    verts_file = nons_file['verts']
    trans_file = nons_file['trans']

    seq_path = os.path.join(datasetDir, 'sequenceFiles/'+folder, seq_name+'.pkl')
    actor = actor
    frame = 0
    
    seq = pkl.load(open(seq_path, 'rb'), encoding='latin1')
    gender = seq['genders'][actor]

    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    input_img_path = os.path.join(datasetDir,'imageFiles',seq_name+'/image_{:05d}.jpg'.format(frame))

    input_img = cv2.imread(input_img_path)
    resize_factor = 2
    input_img = cv2.resize(input_img, (int(input_img.shape[1]/resize_factor), int(input_img.shape[0]/resize_factor)))
    render_image_size = max(input_img.shape[0], input_img.shape[1])
    
    cam_intrinsics = seq['cam_intrinsics']/resize_factor # resize
    half_max_length = max(cam_intrinsics[0:2,2]) 
    principal_point = [-(cam_intrinsics[0,2]-input_img.shape[1]/2.)/(input_img.shape[1]/2.), -(cam_intrinsics[1,2]-input_img.shape[0]/2.)/(input_img.shape[0]/2.)] # - or +
    principal_point = torch.tensor(principal_point).unsqueeze(0)
    f = torch.tensor([(cam_intrinsics[0,0]/half_max_length).astype(np.float32), (cam_intrinsics[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)

    renderer = SMPLRenderer(batch_size, render_image_size, f, 
                            principal_point=principal_point, gender=gender).cuda()    
    output_dir = os.path.join(root, 'test_data/result', seq_name+'_'+str(actor))
    output_img_dir = os.path.join(output_dir, 'front_view')


    results_v = []
    results_t = []
    images = []
    output_verts = []
    output_trans = []

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    
    for frame in trange(0, seq['cam_poses'].shape[0]):

        input_img_path = os.path.join(datasetDir,'imageFiles', seq_name+'/image_{:05d}.jpg'.format(frame))
        input_img = cv2.imread(input_img_path)
        input_img = cv2.resize(input_img, (int(input_img.shape[1]/resize_factor), int(input_img.shape[0]/resize_factor)))

        verts = verts_file[frame]
        trans = trans_file[frame]

        results_v.append(verts)
        results_t.append(trans)
        images.append(input_img)

        # smoothing post-processing
        if frame == 2:
            avg_res = (results_v[0] + results_v[1] + results_v[2]) / 3.0
            output_verts.append(avg_res)
            output_verts.append(avg_res)

            avg_trans = (results_t[0] + results_t[1] + results_t[2]) / 3.0
            output_trans.append(avg_trans)
            output_trans.append(avg_trans)

            rendered_img, _ = renderer.forward(torch.tensor(avg_res).cuda().unsqueeze(0), mode='vis')
            if images[0].shape[0] < images[0].shape[1]:
                rendered_img = rendered_img[0, 210:750,...].cpu().numpy() * 255
            else:
                rendered_img = rendered_img[0, :,210:750].cpu().numpy() * 255
            valid_mask = (rendered_img[:,:,-1] > 0)[:, :, np.newaxis]
            output_img = (rendered_img[:, :, :-1] * valid_mask + images[0] * (1 - valid_mask)).astype(np.uint8)
            cv2.imwrite(output_img_dir+'/{:04d}.png'.format(frame-2),output_img)

            output_img = (rendered_img[:, :, :-1] * valid_mask + images[1] * (1 - valid_mask)).astype(np.uint8)
            cv2.imwrite(output_img_dir+'/{:04d}.png'.format(frame-1),output_img)

        elif frame == (seq['cam_poses'].shape[0] - 1):
            avg_res = (results_v[2] + results_v[3] + results_v[4]) / 3.0
            output_verts.append(avg_res)
            output_verts.append(avg_res)
            
            avg_trans = (results_t[2] + results_t[3] + results_t[4]) / 3.0
            output_trans.append(avg_trans)
            output_trans.append(avg_trans)

            rendered_img, _ = renderer.forward(torch.tensor(avg_res).cuda().unsqueeze(0), mode='vis')
            if images[3].shape[0] < images[3].shape[1]:
                rendered_img = rendered_img[0, 210:750,...].cpu().numpy() * 255
            else:
                rendered_img = rendered_img[0, :,210:750].cpu().numpy() * 255
            valid_mask = (rendered_img[:,:,-1] > 0)[:, :, np.newaxis]
            output_img = (rendered_img[:, :, :-1] * valid_mask + images[3] * (1 - valid_mask)).astype(np.uint8)

            cv2.imwrite(output_img_dir+'/{:04d}.png'.format(frame-1),output_img)

            output_img = (rendered_img[:, :, :-1] * valid_mask + images[4] * (1 - valid_mask)).astype(np.uint8)
            cv2.imwrite(output_img_dir+'/{:04d}.png'.format(frame),output_img)

        if len(results_v) == 5:
            avg_res = (results_v[0] + results_v[1] + results_v[3] + results_v[4]) * 0.15 + results_v[2] * 0.4
            output_verts.append(avg_res)

            avg_trans = (results_t[0] + results_t[1] + results_t[3] + results_t[4]) * 0.15 + results_t[2] * 0.4
            output_trans.append(avg_trans)

            rendered_img, _ = renderer.forward(torch.tensor(avg_res).cuda().unsqueeze(0), mode='vis')
            if images[2].shape[0] < images[2].shape[1]:
                rendered_img = rendered_img[0, 210:750,...].cpu().numpy() * 255
            else:
                rendered_img = rendered_img[0, :,210:750].cpu().numpy() * 255
            valid_mask = (rendered_img[:,:,-1] > 0)[:, :, np.newaxis]
            output_img = (rendered_img[:, :, :-1] * valid_mask + images[2] * (1 - valid_mask)).astype(np.uint8)

            cv2.imwrite(output_img_dir+'/{:04d}.png'.format(frame-2),output_img)
            
            results_v.pop(0)
            results_t.pop(0)
            images.pop(0)

    output_dict = {'verts': output_verts, 'trans': output_trans}
    pkl.dump(output_dict, open(os.path.join(output_dir, 'output_s.pkl'), 'wb'))


