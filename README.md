# Human Performance Capture from Monocular Video in the Wild
## [Paper](https://arxiv.org/pdf/2111.14672.pdf) | [Video](https://www.youtube.com/watch?v=5M7Ytnxmhd4) | [Project Page](https://ait.ethz.ch/projects/2021/human-performance-capture)

<img src="assets/teaser.gif" width="1000" height="350"/> 

Official code release for 3DV 2021 paper [*Human Performance Capture from Monocular Video in the Wild*](https://arxiv.org/pdf/2111.14672.pdf). We propose a method capable of capturing the dynamic 3D human shape from a monocular video featuring challenging body poses, without any additional input.

If you find our code or paper useful, please cite as
```
@inproceedings{guo2021human,
  title={Human Performance Capture from Monocular Video in the Wild},
  author={Guo, Chen and Chen, Xu and Song, Jie and Hilliges, Otmar},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  pages={889--898},
  year={2021},
  organization={IEEE}
}
```

## Quick Start
CLone this repo:
```
git clone https://github.com/MoyGcc/hpcwild.git
cd  hpcwild
conda env create -f environment.yml
conda activate hpcwild
```
Additional Dependencies:
1. Kaolin 0.1.0 (https://github.com/NVIDIAGameWorks/kaolin)
2. MPI mesh library (https://github.com/MPI-IS/mesh)
3. torch-mesh-isect (https://github.com/vchoutas/torch-mesh-isect)

Download [SMPL models](https://smpl.is.tue.mpg.de/downloads) (1.0.0 for Python 2.7 (10 shape PCs)) and move them to the corresponding places:
```
mkdir lib/smpl/smpl_model/
mv /path/to/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl smpl_rendering/smpl_model/SMPL_FEMALE.pkl
mv /path/to/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl smpl_rendering/smpl_model/SMPL_MALE.pkl
```

Download checkpoints for external modules:
```
wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth
mv /path/to/checkpoint_iter_370000.pth external/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth

wget https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt pifuhd.pt 
mv /path/to/pifuhd.pt external/pifuhd/checkpoints/pifuhd.pt

Download IPNet weights: https://datasets.d2.mpi-inf.mpg.de/IPNet2020/IPNet_p5000_01_exp_id01.zip
unzip IPNet_p5000_01_exp_id01.zip
mv /path/to/IPNet_p5000_01_exp_id01 registration/experiments/IPNet_p5000_01_exp_id01

gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O modnet_photographic_portrait_matting.ckpt
mv /path/to/modnet_photographic_portrait_matting.ckpt external/MODNet/pretrained/modnet_photographic_portrait_matting.ckpt
```
### Test on 3DPW dataset
Download [3DPW dataset](https://virtualhumans.mpi-inf.mpg.de/3DPW/) 
1. modify the `dataset_path` in `test.conf`.
2. run `bash mesh_recon.sh` to obtain the rigid body shape.
3. run `bash registration.sh` to register a SMPL+D model to the rigid human body.
4. run `bash tracking.sh` to capture the human performance temporally.

Please refer to [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [MODNet](https://github.com/ZHKKKe/MODNet) to extract 2D joints and masks if you want to test on more sequences.

# Acknowledgement
We use the code in [PIFuHD](https://github.com/facebookresearch/pifuhd) for the rigid body construction and adapt [IPNet](https://github.com/bharat-b7/IPNet) for human model registration. We use off-the-shelf methods [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [MODNet](https://github.com/ZHKKKe/MODNet) for the extraction of 2D keypoints and sihouettes. We sincerely thank these authors for their awesome work.
