import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import os, cv2, sys
from pyhocon import ConfigFactory

from external.pifuhd.apps.recon import recon
from external.pifuhd.lib.options import BaseOptions
from external.MODNet.src.models.modnet import MODNet

sys.path.append('./external/lightweight-human-pose-estimation.pytorch')
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose
import demo

from PIL import Image

def get_rect(net, images, height_size):
    net = net.eval()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts

    for image in images:
        rect_path = image.replace('.%s' % (image.split('.')[-1]), '_rect.txt')
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        rects = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            valid_keypoints = []
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
            valid_keypoints = np.array(valid_keypoints)

            if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
              pmin = valid_keypoints.min(0)
              pmax = valid_keypoints.max(0)

              center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
              radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))
              
            elif pose_entries[n][10] == -1.0 and pose_entries[n][13] == -1.0 and pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
              # if leg is missing, use pelvis to get cropping
              center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int)
              radius = int(1.45*np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0))
              center[1] += int(0.05*radius)
              
            else:
              center = np.array([img.shape[1]//2,img.shape[0]//2])
              radius = max(img.shape[1]//2,img.shape[0]//2)
              

            x1 = center[0] - radius
            y1 = center[1] - radius
            
            rects.append([x1, y1, 2*radius, 2*radius])
        if len(rects) != 0:
          np.savetxt(rect_path, np.array(rects), fmt='%d')

def mesh_reconstruction_modnet(im=None, input_img=None, seg_actor=0, segmentation_tool=None, name=None, frame=0, threshold=20):
    _, _, matte = segmentation_tool(im.cuda(), True)
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')

    gt_silhouettes = matte[seg_actor].data *255
    gt_valid_mask = (gt_silhouettes > threshold).squeeze(0).cpu().numpy()[:,:,np.newaxis]
    segmented_img = input_img * gt_valid_mask

    cv2.imwrite('./test_image/test_image.jpg', segmented_img)
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('./external/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth', map_location='cpu')
    load_state(net, checkpoint)
    get_rect(net.cuda(), ['./test_image/test_image.jpg'], 512)

    start_id = -1
    end_id = -1
    recon_parser = BaseOptions()
    cmd = ['--dataroot', './test_image',  
           '--results_path', './results', 
           '--loadSize', '1024', 
           '--resolution', '256', 
           '--load_netMR_checkpoint_path', './external/pifuhd/checkpoints/pifuhd.pt',
           '--start_id', '%d' % start_id, 
           '--end_id', '%d' % end_id]
    opt = recon_parser.parse(cmd)
    recon(opt, name, frame, True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    
    args = parser.parse_args()
    conf = args.config
    conf = ConfigFactory.parse_file(conf)
    dataset_path = conf.get_string('dataset_path') 
    dataset = conf.get_string('dataset')
    seq = conf.get_string('seq')
    frame_num = conf.get_int('frame_num')
    threshold = conf.get_float('seg_threshold')
    if dataset == '3DPW':
        image_file_path = os.path.join(dataset_path, 'imageFiles')

    ref_size = 512
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load('./external/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt'))
    modnet.eval()
    

    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    image_path = os.path.join(image_file_path, seq, 'image_%05d.jpg' % frame_num)
    input_img = cv2.imread(image_path)
    im = Image.open(image_path)

    im = im_transform(im)
    im = im[None, :, :, :]
    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w     
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    mesh_reconstruction_modnet(im, input_img, 0, modnet, dataset+'_'+seq, frame_num, threshold)