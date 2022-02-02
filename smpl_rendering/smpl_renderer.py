import torch
import torch.nn as nn
import numpy as np
import os
from .smplx.body_models import SMPL

from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    Textures,
    Materials,
    look_at_view_transform
)
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import TexturesUV

materials = Materials(ambient_color=((0.7, 0.7, 0.7),), device='cuda')

from typing import NamedTuple, Sequence

class SMPLBlendPar(NamedTuple):
    sigma: float = 1e-9 # 0 sys.float_info.epsilon
    gamma: float = 1e-4
    background_color: Sequence = (1.0, 1.0, 1.0)

class SMPLRenderer(nn.Module):
    def __init__(self, batch_size, image_size, f=50, principal_point=((0.0, 0.0),), R=None, t=None, model_path=None, gender='female'):

        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.f = f
        self.principal_point = principal_point
        if model_path is None:
            model_path = os.path.join(os.getcwd(), 'smpl_rendering/smpl_model')
        self.smpl_mesh = SMPL(model_path=model_path, batch_size = batch_size, gender=gender)

        self.v_template = self.smpl_mesh.v_template.cuda()
        self.blend_params = SMPLBlendPar()

        if R is None and t is None: 
            self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                    [0., -1., 0.],
                                                    [0., 0., 1.]])).cuda().float().unsqueeze(0)

            
            self.cam_T = torch.zeros((1,3)).cuda().float()

            # no adaption to PyTorch3D needed
        else:
            # using the 'cam_poses' from 3DPW
            self.cam_R = R.detach().clone()
            self.cam_T = t.detach().clone()     

            # coordinate system adaption to PyTorch3D
            self.cam_R[:, :2, :] *= -1.0
            self.cam_T[:, :2] *= -1.0
            self.cam_R = torch.transpose(self.cam_R,1,2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        torch.cuda.set_device(self.device)

        self.cameras = SfMPerspectiveCameras(focal_length=self.f, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)

        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=10,blur_radius=0)

        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.faces = torch.from_numpy(self.smpl_mesh.faces.astype(int)).unsqueeze(0).cuda()

        self.mask_shader = SoftSilhouetteShader(blend_params=self.blend_params)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        tex_lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), device=self.device)
        tex_lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                                 ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
        self.vis_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftPhongShader(
                                                                                            device=self.device, 
                                                                                            cameras=self.cameras,
                                                                                            lights=lights,
                                                                                            materials=materials
                                                                                           ))
        self.tex_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftPhongShader(
                                                                                            device=self.device, 
                                                                                            cameras=self.cameras,
                                                                                            lights=tex_lights
                                                                                           ))        
        self.mask_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.mask_shader)

        verts_rgb = torch.ones_like(self.v_template.unsqueeze(0)).cuda()

        self.mask_texture = TexturesVertex(verts_features=verts_rgb)

        _, faces, aux = load_obj('/home/chen/Semester-Project/smpl_rendering/text_uv_coor_smpl.obj', load_textures=True)

        self.verts_uvs = aux.verts_uvs.expand(batch_size, -1, -1).to(self.device)
        self.faces_uvs = faces.textures_idx.expand(batch_size, -1, -1).to(self.device)

    # get the individual T_hip for each person
    def get_T_hip(self, betas=None, displacement=None):
        return self.smpl_mesh.get_T_hip(betas, displacement)
    
    def get_smpl_output(self, betas, thetas, trans, displacement=None, absolute_displacement=False):

        if displacement is not None and absolute_displacement:
            displacement = displacement - self.v_template

        smpl_output = self.smpl_mesh.forward(betas=betas, body_pose=thetas[:,3:],
                                             transl=trans,
                                             global_orient=thetas[:,:3],
                                             displacement=displacement,
                                             return_verts=True)
        return smpl_output
    def side_view(self, smpl_vertices, trans, texture=None):
        elev = torch.linspace(0, 360, 30) 
        azim = torch.tensor([-160]*30)    

        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
        R[:, :2, :] *= -1.0
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1,2)

        camera = OpenGLPerspectiveCameras(device=self.device, R=R[None, 0, ...], T=T[None, 0, ...])

        if texture is not None:
            sideV_lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), device=self.device)
            mesh_texture = TexturesUV(maps=texture, 
                                      faces_uvs=self.faces_uvs,
                                      verts_uvs=self.verts_uvs)
            sideV_mesh = Meshes(smpl_vertices-trans, faces=self.faces.expand(self.batch_size,-1,-1), textures=mesh_texture).cuda()
            sideV_img = self.tex_renderer(sideV_mesh, cameras=camera, lights=sideV_lights)
        else:
            sideV_lights = PointLights(device=self.device, location=[[0.0, 0.0, -2.0]])
            mesh_texture = self.mask_texture
            sideV_mesh = Meshes(smpl_vertices-trans, faces=self.faces.expand(self.batch_size,-1,-1), textures=mesh_texture).cuda()
            sideV_img = self.vis_renderer(sideV_mesh, cameras=camera, lights=sideV_lights)
        
        return sideV_img

    def forward(self, smpl_vertices, mode='a', texture=None):
        '''

        '''        

        if 'a' in mode:
            mask_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()

            mask = self.mask_renderer(mask_mesh)[...,3] 

            return mask, mask_mesh
        elif 'tex' in mode:
            mesh_texture = TexturesUV(maps=texture,
                                      faces_uvs=self.faces_uvs,
                                      verts_uvs=self.verts_uvs)
            texture_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=mesh_texture).cuda()

            texture_img = self.tex_renderer(texture_mesh)
            return texture_img, texture_mesh
        elif 'vis' in mode:

            vis_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()
            rendered_img = self.vis_renderer(vis_mesh)
            return rendered_img, vis_mesh

        elif 'norm' in mode:
            mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1)).cuda()
            normal = mesh.verts_normals_packed()
            normal = -1*(normal) * 0.5 + 0.5
            normal = normal[:,[2,1,0]]
            verts = mesh.verts_packed()[None].cuda()

            colors = normal[None].cuda()
            mesh_texutre= Textures(verts_rgb=colors)
            norm_mesh = Meshes(verts=verts,faces=self.faces.expand(self.batch_size,-1,-1),textures=mesh_texutre).cuda()
            rendered_img = self.tex_renderer(norm_mesh)  
            return rendered_img, norm_mesh      
        else:
            raise NotImplementedError('%s mode not supported by SMPLRenderer yet.' % mode)


