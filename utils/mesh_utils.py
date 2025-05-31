import os
import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f'{abs_path}/third_party/diffusion-net/src']
import diffusion_net

import pickle
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack
import trimesh

from copy import deepcopy
from torch_scatter import scatter_add
from torch_sparse import coalesce, transpose
from cupyx.scipy.sparse.linalg import SuperLU

import pytorch3d

from utils.deformation_transfer import Transfer

def sample_point_in_triangle(batch_v, faces, lamdas=None, normalize=False):
    """
    Args:
        batch_v (torch.tensor): [B, V, 3] batched vertices
        faces (torch.tensor): [F, 3]
        lamdas (torch.tensor): [2,] for interpolation v0-v1 -> tmp, tmp-v2 -> new vertices (optional)
        normalize (bool): if True, normalize (for normal vectors)
    Return:
        new_v_T (torch.tensor): [B, V, 3]
    """
    vf_T = batch_v[:, faces]
    
    if lamdas is None:
        lamdas = torch.rand(2).to(vf_T.device)

    new_v_T = vf_T[:,:,0] * lamdas[0] + vf_T[:,:,1] * (1-lamdas[0])
    new_v_T = new_v_T * lamdas[1] + vf_T[:,:,2] * (1-lamdas[1])
    if normalize:
        new_v_T = torch.nn.functional.normalize(new_v_T, p=2, dim=-1)  # [N, 3]
    return new_v_T
    
def barycentric_coordinate(T, P):
    """
    Args:
        T (torch.tensor): triangle vertices [3, 3]
        P (torch.tensor): point on the triangle [3, ]
    Return:
        lambdas (torch.tensor): [3, ]
    """
    A, B, C = T[0], T[1], T[2]
    mat = torch.tensor([
            [A[0], A[1], A[2], 1],
            [B[0], B[1], B[2], 1],
            [C[0], C[1], C[2], 1],
        ]).T  # T
    P_ext = torch.tensor([P[0], P[1], P[2], 1])
    lambdas = torch.linalg.pinv(mat.float()) @ P_ext.float()
    lambdas = lambdas[:3]
    lambdas[2] = 1- lambdas[0] -lambdas[1]
    return lambdas

def calc_cent(vertices, faces, mode='np'):
    """
    Args:
        vertices (torch.tensor): [B, V, 3] vertices 
        faces (torch.tensor): [F, 3] vertex index for each face
    """
    if len(vertices.size()) < 3:
        vertices = vertices.unsqueeze(0)
    fv = vertices[:, faces]
    
    if mode=='np':
        return np.mean(fv, axis=-2)
    else:
        return torch.mean(fv, dim=-2)

def calc_norm_torch(batch_v, face, at='face'):
    """
    Args:
        batch_v (torch.tensor): [B*T, V, 3] vertices for current batch
        face (torch.tensor): [F, 3] vertex index for each face
        at (str): mode 'face', 'vertex' (default: 'face')
        
    Returns:
        norm | face_norm (torch.tensor): corresponding normal vector
    """
    B_S = batch_v.shape[0]
    N_V = batch_v.shape[1]

    batch_vf = batch_v[:, face] # --> [B, F, 3, 3]
    span = batch_vf[..., 1:, :] - batch_vf[..., :1, :] # --> [B, V, 2, 3]
    cross = torch.linalg.cross(span[..., 0, :], span[..., 1, :], dim=-1) # --> [B, F, 3]
    face_norm = torch.nn.functional.normalize(cross, p=2, dim=-1)  # --> [B, F, 3]
    
    if at == 'face':
        return face_norm
    else: # at == 'vertex'
        idx = torch.cat([face[:, 0], face[:, 1], face[:, 2]], dim=0)
        face_norm = face_norm.repeat(1, 3, 1)

        norm = scatter_add(face_norm, idx, dim=1, dim_size=N_V)
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)  # [N, 3]
        return norm

###### Reference for codes below: https://github.com/dafei-qin/NFR_pytorch/blob/master/myutils.py
def get_dfn_info(mesh, cache_dir=None, device='cuda'):
    """
    Args:
        mesh (trimesh.Trimesh)
    """
    verts_list = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.faces).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(mesh.faces)]
    dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def get_dfn_info2(vertices, faces, cache_dir=None, device='cuda'):
    """
    Args:
        vertices (np.ndarray)
        faces (np.ndarray)
    """
    if type(vertices)==np.ndarray:
        vertices=torch.from_numpy(vertices)
    if type(faces)==np.ndarray:
        faces=torch.from_numpy(faces)
        
    verts_list = vertices.unsqueeze(0).float()
    face_list  = faces.unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], faces]
    dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

def get_span_matrix(batch_vertices, faces):
    """
    Args
        batch_vertices (torch.tensor): [B, V, 3]
        faces (torch.tensor): [F, 3]
    Return
        span (torch.tensor): [B, F, 3, 3] (v2-v1, v3-v1, v4-v1)
    """
    B_v = batch_vertices

    faces  = faces[None].repeat(B_v.shape[0], 1 ,1)
    B, num_faces  = faces.shape[:2]
    batch_indices = torch.arange(B)[:, None, None]
    batch_indices = torch.tile(batch_indices, (1, num_faces, 1))

    B_vf = B_v[batch_indices, faces].permute(0, 1, 3, 2)

    v1, v2, v3 = B_vf[..., 0], B_vf[..., 1], B_vf[..., 2]

    cross = torch.linalg.cross(v2 - v1, v3 - v1, dim=-1)

    vn = torch.nn.functional.normalize(cross, p=2, dim=-1)  # [F, 3]

    v4 = v1 + vn

    span = torch.stack((v2 - v1, v3 - v1, v4 - v1), dim=-1)
    return span

def get_jacobian_matrix(verts, faces, template, return_torch=False):
    """Reference from Deformation Transfer for Triangle Meshes [Sumner and Popovic, 2004]
    Args
        verts (torch.tensor): [B*T, V, 3] target vertices (deformed)
        faces (torch.tensor): [F, 3]
        template (torch.tensor): [B, V, 3] source vertices (undeformed)
    Return
        Q (torch.tensor): [B, F, 3, 3] transformations (v2-v1, v3-v1, v4-v1)
    """
    B, V, _ = verts.shape

    span_matrix = get_span_matrix(verts, faces)
    neutral_span_matrix = get_span_matrix(template, faces)

    # https://pytorch.org/docs/stable/generated/torch.linalg.inv.html -> [Solving A @ X = B (X = A^-1 @ B)]
    # Consider using torch.linalg.solve() if possible for multiplying a matrix on the left by the inverse, as:
    # ``` linalg.solve(A, B) == linalg.inv(A) @ B ```
    # When B is a matrix

    # It is always preferred to use `solve()` when possible, 
    # as it is faster and more numerically stable than computing the inverse explicitly.

    # It is possible to compute the solution of the system X @ A = B (X = B @ A^-1) 
    # by passing the inputs A and B transposed and transposing the output returned by this function.
    # ``` linalg.solve(A.T, B.T).T == B @ linalg.inv(A) ```

    # neutral_span_inv_matrix = torch.linalg.inv(neutral_span_matrix)
    # Q = (span_matrix @ neutral_span_inv_matrix).permute(0, 1, 3, 2)
    Q = torch.linalg.solve(neutral_span_matrix.permute(0, 1, 3, 2), span_matrix.permute(0, 1, 3, 2))
    return Q

class Normalizer(object):
    def __init__(self, std_path, device, zero_mean=True):
        if zero_mean:
            self.gradients_std = np.load(os.path.join(std_path, 'gradients_std.npy'))
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.eye(3).view(-1,).unsqueeze(0).unsqueeze(0).to(device).float()
        else:
            self.gradients_mean, self.gradients_std = np.load(os.path.join(std_path, 'wks_mean_std.npz'), allow_pickle=True)['arr_0'].item().values()
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.from_numpy(self.gradients_mean).to(device).float()

    def normalize(self, tensor):
        return (tensor - self.gradients_mean.to(tensor.device)) / self.gradients_std.to(tensor.device)

    def inv_normalize(self, tensor):
        return tensor * self.gradients_std.to(tensor.device) + self.gradients_mean.to(tensor.device)
    
class Normalizer_img(Normalizer):
    def __init__(self, std_path, device):
        import warnings
        warnings.filterwarnings('ignore')
        with open(os.path.join(std_path, 'img_stat.pkl'), 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()

def mesh_transform(mesh,
                   mesh_data="voca",
                   util_dir=f"{abs_path}/mesh_utils",
                   inverse=False):
    """
    Args:
        mesh: trimesh
        mesh_data: the type of mesh data ["biwi", "voca"]
    Return
        mesh
    """
    # load
    mat = np.load(f'{util_dir}/{mesh_data}/align.npy')

    # clone mesh
    mesh_new = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    
    # pre-process biwi
    if mesh_data == 'biwi': 
        if not inverse:
            # for BIWI, mean vertex needs to be moved to zero center
            mesh_new.vertices = mesh_new.vertices - mesh_new.vertices.mean(axis=0) # - np.array([0, 0, 0.03])

    # make 4x4 matrix
    if mat.shape == (3, 4):
        tmp_identity = np.eye(4)[3:4,:]
        mat = np.concatenate([mat, tmp_identity], axis=0)
        # print("voca_mat: ", mat)
    
    # make inverse
    if inverse:
        mat = np.linalg.inv(mat)

    # transform
    mesh_v = np.hstack((mesh_new.vertices, np.ones((mesh_new.vertices.shape[0], 1))))
    mesh_new_v = np.dot(mesh_v, mat.T)
    mesh_new.vertices = mesh_new_v[:,:3]
    return mesh_new

def vert_transform_batch(vertices,
                   mesh_data="voca",
                   util_dir=f"{abs_path}/mesh_utils",
                   inverse=False):
    """
    Args:
        vertices (np.array): [V, 3] vertices of the mesh
        mesh_data: the type of mesh data ["biwi", "voca"]
    Return
        mesh
    """
    # load
    mat = np.load(f'{util_dir}/{mesh_data}/align.npy')
    
    # pre-process biwi
    if mesh_data == 'biwi': 
        if not inverse:
            # for BIWI, mean vertex needs to be moved to zero center
            vertices = vertices - vertices.mean(axis=0) # - np.array([0, 0, 0.03])

    # make 4x4 matrix
    if mat.shape == (3, 4):
        tmp_identity = np.eye(4)[3:4,:]
        mat = np.concatenate([mat, tmp_identity], axis=0)
        # print("voca_mat: ", mat)
    
    # make inverse
    if inverse:
        mat = np.linalg.inv(mat)

    # transform
    mesh_v = np.concatenate((vertices, np.ones((*vertices.shape[:-1], 1))), axis=-1)
    mesh_new_v = np.dot(mesh_v, mat.T)
    return mesh_new_v[...,:3]

def get_mesh_operators(mesh):
    N_FACE = mesh.faces.shape[0]
    N_VERTEX = mesh.vertices.shape[0]
    transf = Transfer(mesh, deepcopy(mesh))
    lu_solver = SuperLU(transf.lu)
    idxs, vals = coalesce(from_dlpack(transf.idxs.toDlpack()).long(), from_dlpack(transf.vals.toDlpack()), m=N_FACE *3, n=N_VERTEX)
    idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)
    rhs = transf.cupy_A.T
    return lu_solver, idxs, vals, rhs

class Renderer:
    def __init__(self, view_d=6, img_size=1024, fragments=False):

        import warnings
        warnings.filterwarnings('ignore')
        from pytorch3d.renderer import (
        look_at_view_transform,
        FoVPerspectiveCameras, 
        PointLights, 
        Materials, 
        RasterizationSettings, 
        MeshRenderer,
        MeshRendererWithFragments,
        MeshRasterizer)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        R, T = look_at_view_transform(view_d, 0, 0) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
            cull_backfaces=True
        )
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        self.return_fragment = fragments
        if self.return_fragment:
            rd = MeshRendererWithFragments
        else:
            rd = MeshRenderer
        materials = Materials(
            device=device,
            specular_color=[[0.0, 0.0, 0.0]],
            shininess=100
        )
        # color = [172, 219, 255]
        color = torch.tensor([255, 255, 255]) / 2 / 255
        lights = PointLights(device=device, location=[[0.0, 0.0, 6]])
        renderer = rd(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.HardFlatShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )

        self.renderer = renderer
        self.color = color
        self.device = device
        self.materials = materials
        self.img_normalizer = Normalizer_img(f'{abs_path}/data/MF_all_v5', 'cuda:0')

    def renderbatch(self, vertices, faces, reverse=False):
        meshes = pytorch3d.structures.Meshes(verts=vertices, faces=faces)
        meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).unsqueeze(0).expand(len(meshes), meshes.num_verts_per_mesh()[0], -1))
        if self.return_fragment:
            images, fragments = self.renderer(meshes,materials=self.materials)
        else:
            images = self.renderer(meshes,materials=self.materials)
            fragments = None
        if reverse:
            return 1 - images[..., :3], fragments
        return images[..., :3], fragments


    def mesh2img(self, mesh, reverse=False, noise=False):
        mesh = pytorch3d.structures.Meshes(verts=[torch.from_numpy(mesh.vertices).float().to(self.device)], faces=[torch.from_numpy(mesh.faces).to(self.device)])
        mesh.textures = pytorch3d.renderer.TexturesVertex(verts_features=torch.tensor(self.color).to(self.device).unsqueeze(0).expand_as(mesh.verts_packed())[None])
        if self.return_fragment:
            images, fragments = self.renderer(mesh,materials=self.materials)
        else:
            images = self.renderer(mesh,materials=self.materials)
            fragments = None
        if noise:
            images += (torch.randn(images.shape[:3]).unsqueeze(-1) * 0.01).to(images.device)
        if reverse:
            return ((1 - images[0, ..., :3].detach().cpu().numpy()) * 255).astype(np.uint8), fragments
        return (images[0, ..., :3].detach().cpu().numpy() * 255).astype(np.uint8), fragments
    
    def render_img(self, mesh, img_enc='cnn', img_path=''):
        """
        Args:
            mesh (trimesh.Trimesh): triangle mesh
            img_enc (str): encoder type
            img_path (str): save path for rendered image
        Return:
            img (torch.tensor): rendered image [1, 256, 256, 3]
        """
        img, fragments = self.renderbatch(
            [torch.from_numpy(mesh.vertices).float().to(self.device)], 
            [torch.from_numpy(mesh.faces).float().to(self.device)], 
            reverse=True
        )
        zbuf = fragments.zbuf[0, ..., 0].float()
        img = torch.cat((img, zbuf.unsqueeze(0).unsqueeze(-1)), dim=-1)
        img[img == -1] = img.amax(dim=(0, 1, 2))[-1] * 2
        img = self.img_normalizer.normalize(img)
        if img_path != '':
            np.save(img_path, img.cpu().numpy())
        if img_enc == 'cnn':
            img = img[..., :3]
        return img