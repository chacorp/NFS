# import potpourri3d as pp3d
import trimesh
import open3d as o3d
from copy import deepcopy
import numpy as np
from numpy.linalg import eigvals
# from scipy.linalg import polar
from scipy.spatial.transform import Rotation
import torch
# try:
import pytorch3d
from pytorch3d import transforms
from torch_scatter import scatter_add
import transforms3d as t3d
import scipy
# import scipy.sparse.linalg as sla
import os
# import warnings
import igl
# from mesh_signatures.signature import SignatureExtractor
import sys
import pickle
# from tqdm import tqdm

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f"{abs_path}/third_party/diffusion-net/src"]
import diffusion_net



class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.edges = None
        self.vertex_degree = None
        self.face_adjacency_edges = None
        self.face_adjacency_unshared = None
        # self.edges_unique_length = None
        self.edges_unique = None
        # self.area_faces = None

        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)

    def __getitem__(self, key):
        if key == 0:
            return self.vertices

        elif key == 1:
            return self.faces
        else:
            raise KeyError # print(f"Only allows Key 0: vertices, Key 1: faces, but receives Key {key} ")

    def update_area(self):
        target_fv = self.vertices[self.faces]
        AB = target_fv[:, 1] - target_fv[:, 0]
        AC = target_fv[:, 2] - target_fv[:, 0]
        self.area_faces = 0.5 * np.linalg.norm(np.cross(AB, AC), axis=-1)

    @classmethod
    def load(self, path: str, read_face=True):
        """
        Load obj file
        load the .obj format mesh file with square or triangle faces
        return the vertices list and faces list
        """
        if path.endswith('.obj'):
            file = open(path, 'r')
            lines = file.readlines()
            vertices = []
            faces = []
            for line in lines:
                if line.startswith('v') and not line.startswith('vt') and not line.startswith('vn'):
                    line_split = line.split(" ")
                    # ver = line_split[1:4]
                    ver = [each for each in line_split[1:] if each != '']
                    ver = [float(v) for v in ver]
                    vertices.append(ver)
                else:
                    if read_face:
                        if line.startswith('f'):
                            line_split = line.split(" ")
                            if '/' in line:
                                tmp_faces = line_split[1:]
                                f = []
                                if '\n' in tmp_faces:
                                    tmp_faces.pop(tmp_faces.index('\n'))
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face.split('/')[0]))
                                faces.append(f)
                            else:
                                tmp_faces = line_split[1:]
                                f = []
                                for tmp_face in tmp_faces:
                                    f.append(int(tmp_face))
                                faces.append(f)
                    else:
                        pass

            if read_face:
                file.close()
                return Mesh(np.array(vertices), np.array(faces) - 1)
            else:
                file.close()
                return Mesh(np.array(vertices), None)
        # else:
        #     raise ValueError('Wrong file format, not a correct .obj mesh file!')
        #     ret

    @classmethod
    def from_trimesh(self, mesh: trimesh.Trimesh):
        
        new_mesh = Mesh(deepcopy(mesh.vertices), deepcopy(mesh.faces))

        new_mesh.edges = deepcopy(mesh.edges)
        new_mesh.vertex_degree = deepcopy(mesh.vertex_degree)
        new_mesh.face_adjacency_edges = deepcopy(mesh.face_adjacency_edges)
        new_mesh.face_adjacency_unshared = deepcopy(mesh.face_adjacency_unshared)
        # new_mesh.edges_unique_length = deepcopy(mesh.edges_unique_length)
        new_mesh.edges_unique = deepcopy(mesh.edges_unique)
        # new_mesh.area_faces = deepcopy(mesh.area_faces)
        
        return new_mesh


    
    def transfer(self, shift, scale):
        self.vertices = self.vertices * np.array(scale)[np.newaxis]
        self.vertices = self.vertices + np.array(shift)[np.newaxis]
        self.update_area()
        
    def write(self, file_name_path):
        faces = self.faces
        vertices = self.vertices
        faces = faces + 1
        with open(file_name_path, 'w') as f:
            for v in vertices:
                # print(v)
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            for face in faces:
                if len(face) == 4:
                    f.write("f {} {} {} {}\n".format(face[0], face[1], face[2], face[3])) 
                if len(face) == 3:
                    f.write("f {} {} {}\n".format(face[0], face[1], face[2])) 


def calc_norm(mesh):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def convert_per_face_to_per_vertex(inputs_f, face, n_vertex):

    # use a sparse matrix of which face contains each vertex to
    # figure out the summed normal at each vertex
    # allow cached sparse matrix to be passed

    indices = np.asanyarray(face)
    columns = n_vertex

    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape(
        (-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    data = np.ones(len(col), dtype=bool)

    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)),
                                    shape=shape,
                                    dtype=data.dtype)



    summed = torch.from_numpy(matrix.dot(inputs_f.transpose(0, 1).reshape(inputs_f.shape[1], -1)).reshape(matrix.shape[0], inputs_f.shape[0], inputs_f.shape[-1]).transpose(1, 0, 2))
    
    return summed
        
def face_to_vertex_torch(face):
    face = face.numpy()
    indices = face
    columns = np.max(face) + 1
    row = indices.reshape(-1)
    col = np.tile(np.arange(len(indices)).reshape(
        (-1, 1)), (1, indices.shape[1])).reshape(-1)

    shape = (columns, len(indices))
    data = np.ones(len(col), dtype=bool)

    # assemble into sparse matrix
    matrix = scipy.sparse.coo_matrix((data, (row, col)))
    coo = matrix.multiply(1 / (matrix.sum(axis=1) + 1e-6))
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return matrix                            

def calc_cent(mesh):
    fv = mesh.vertices[mesh.faces]
    return fv.mean(axis=-2) 

def get_biggest_connected(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))

    triangle_clusters, cluster_n_triangles, cluster_area = (o3d_mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    o3d_mesh_1 = deepcopy(o3d_mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    o3d_mesh_1.remove_triangles_by_mask(triangles_to_remove)

    return Mesh(np.asarray(o3d_mesh_1.vertices), np.asarray(o3d_mesh_1.triangles))

def remove_degenerated_triangles(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.faces))
    o3d_mesh_removed = o3d_mesh.remove_degenerate_triangles()
    return Mesh(np.asarray(o3d_mesh_removed.vertices), np.asarray(o3d_mesh_removed.triangles))

def calc_jacobian_stat(jacobians):
    """
    Args:
        jacobians: (N, 3, 3) N matrices of the jacobians
    
        # Using the Polar Decomposition J = UP
         - U: Unitary, rotation
         - P: Positive semi-definite Hermitian (Symmetric) Matrix
    Return:
        theta: Rotation angle
        omega: Rotation axis
        A: scaling factor
        
    """
    W, S, V = np.linalg.svd(jacobians, full_matrices=True)
    P = (V.transpose(0, 2, 1) * S[:, np.newaxis]) @ V # Scaling matrix
    U = W @ V # Rotation matrix
    A = eigvals(P)
    rotvec = Rotation.from_matrix(U).as_rotvec()

    theta = np.linalg.norm(rotvec, axis=-1)
    omega = rotvec / (theta[:, np.newaxis] + 1e-5)


    return theta, omega, A

def apply_deformation(triangles, jacobians):
    # triangles: (N, 3, 3) N triangles where (i, :, j) is the jth vertex of the ith triangle
    # jacobians: (N, 3, 3) N deformation gradients
    fix_point = triangles.mean(axis=1, keepdims=True)
    # fix_point = 0
    
    triangles_shift = triangles - fix_point
    
    spans_t = jacobians @ triangles_shift
    return spans_t + fix_point
    
def quat2exp(quants):

    if len(quants.shape) ==3:
        sp = quants.shape[:-1]
        quants = quants.reshape(-1, 4)
    else:
        sp = None
    results = np.array([(t3d.euler.quat2axangle(q)) for q in quants])
    results_rot = Rotation.from_euler('xyz', np.array([t3d.euler.axangle2euler(v, the) for (v, the) in results])).as_matrix()

    results = np.array([a*b for (a, b) in results])
    if sp:
        return results.reshape((*sp, 3)), results_rot.reshape((*sp, 3, 3))
    else:
        return results, results_rot

def decompose_jacobians(jacobians, repr='6dof'):
    # jacobians: (B, N, 9) B x N matrices of the jacobians
    # J = UP
    # U: Rotations, representated by the 6dof
    # P: Scaling, representated by a positive semi-definite symmetric matrix
    
    B = jacobians.shape[0]
    N = jacobians.shape[1]
    jacobians = jacobians.reshape(-1, 3, 3)
    jacobians = jacobians.numpy()
    W, S, V = np.linalg.svd(jacobians, full_matrices=True)
    P = (V.transpose(0, 2, 1) * S[:, np.newaxis]) @ V # Scaling matrix
    U = W @ V # Rotation matrix
    U = torch.from_numpy(U)
    P = torch.from_numpy(P)

    if repr == '6dof':
        rots = transforms.matrix_to_rotation_6d(U)
    elif repr == 'quat':
        rots = transforms.matrix_to_quaternion(U)
    elif repr == 'expmap':
        rots = transforms.matrix_to_axis_angle(U)


    scals = P.reshape(-1, 9)[:, [0, 1, 2, 4, 5, 8]]


    return torch.cat((rots.reshape(B, N, -1),  scals.reshape(B, N, -1)), dim=-1)

def reconstruct_jacobians(inputs, repr='matrix', eps=1e-9):
    # inputs: (BS, N, ?) inputs for reconstruct the jacobians
    # ? = 9, repr == 'matrix'
    # ? = 12, repr == '6dof'
    # ? = 10, repr == 'quat'
    # ? = 9,  repr == 'expmap'
    BS = inputs.shape[0]
    N = inputs.shape[1]
    if repr == 'matrix':
        return inputs.reshape(BS, N, 3, 3)
        
    rots = inputs[:, :, :-6]
    scals = inputs[:, :, -6:]
    # rots = rots.reshape(BS, N, 2, 3)
    rot_matrix = torch.empty(BS, N, 3, 3)

    if repr == '6dof':
        rot_matrix = transforms.rotation_6d_to_matrix(rots)
    elif repr == 'quat':
        rot_matrix = transforms.quaternion_to_matrix(rots)
    elif repr == 'expmap':
        rot_matrix = transforms.axis_angle_to_matrix(rots)

    scal_matrix = torch.empty((BS, N, 9)).to(inputs.device)
    scal_matrix[:, :, [0, 1, 2, 4, 5, 8]] = scals
    scal_matrix[:, :, [3, 6, 7]] = scals[:, :, [1, 2, 4]]
    scal_matrix = scal_matrix.reshape(BS, N, 3, 3)

    return torch.matmul(rot_matrix, scal_matrix)



def render_img(mesh, renderer, img_normalizer, img_enc='cnn', img_path='', use_p2f=False):
    if renderer is not None: # Render the image and save
        img, fragments = renderer.renderbatch(
            [torch.from_numpy(mesh.vertices).float().to('cuda')], 
            [torch.from_numpy(mesh.faces).float().to('cuda')], 
            reverse=True
        )
        zbuf = fragments.zbuf[0, ..., 0].float()
        img = torch.cat((img, zbuf.unsqueeze(0).unsqueeze(-1)), dim=-1)
        img[img == -1] = img.amax(dim=(0, 1, 2))[-1]  * 2
        img = img_normalizer.normalize(img)
        if img_enc == 'cnn':
            img = img[..., :3]
        if use_p2f:
            p2f = fragments.pix_to_face[0, ..., 0].cpu().float()
            mapping = torch.zeros((mesh.faces.shape[0], p2f.shape[0]**2))
            for i in range(mapping.shape[0]):
                mapping[i][p2f.reshape(-1) == i] = 1
                if mapping[i].max() != 0:
                    mapping[i] /= torch.sum(mapping[i])
            mapping = mapping.float().to_sparse()
        np.save(img_path, img.cpu().numpy())
        return img
    else: # Load the image if needed
        if len(img_path) > 0:
            img = torch.from_numpy(np.load(img_path))
        else:
            img = None
        exit("[ERROR][util_nfr.render_img] No renderer is provided")
    return img

def generate_new_z_(model, vertices, faces, renderer, img_normalizer, dfn_info):
    """
    Args:
        model (nn.Module):      pretrained NFR model
        vertices (torch.tensor):[B, V, 3] source mesh with expression
        faces (torch.tensor):   [F, 3] source mesh faces
        renderer:               myutils.renderer
        img_normalizer:         myutils.img_normalizer
        dfn_info:               diffusionNet info extracted from the source/target mesh with neutral face \
                                :: need to recalculate when source mesh changes ::                                
    Return:
        z (torch.tensor):       expression code
    """
    B, V, _ = vertices.shape

    # img
    vertices = vertices.float()
    #img_ = render_img_list([v for v in vertices], [faces]*B, renderer, img_normalizer, img_enc='')
    img_ = render_img([v for v in vertices], [faces]*B, renderer, img_normalizer, img_enc='')

    # vert&norm
    from mesh_utils import calc_norm_torch
    normals  = calc_norm_torch(vertices, faces, at='vert').to(vertices.device)
    inputs_v = torch.cat([vertices, normals], dim=-1)

    with torch.no_grad():
        model.update_precomputes(dfn_info)
        z = model.encode(inputs_v.to(vertices.device), img_.to(vertices.device), N_F=faces.shape[0])
    return z

def generate_new_z(args,
                   model,
                   dataset,
                   face,
                   idx,
                   dfn_info,
                   calc_dfn_info=False,
                   scale=1,
                   shift=np.zeros((1, 3)),
                   offset = np.zeros((1, 53))):
    # global dataset, face, args, latent_dim
    img_, f2v_t_idxs, f2v_t_vals, p2f_t_idxs, p2f_t_vals = None, None, None, None, None
    gradX, gradY = None, None
    if args.img:
        img_ = dataset.img[idx].to(args.device).unsqueeze(0)
        if args.pix2face:
            # Construct sparse matrices
            f2v_t_idxs = dataset.face2vertex_indices.to(args.device).unsqueeze(0)
            f2v_t_vals = dataset.face2vertex_values.to(args.device).unsqueeze(0)
            p2f_t_idxs = dataset.pix2face_indices[idx].to(args.device).unsqueeze(0)
            p2f_t_vals = dataset.pix2face_values[idx].to(args.device).unsqueeze(0)
    if dataset.gradX is not None:
        gradX = dataset.gradX[idx].unsqueeze(0).to(args.device)
        gradY = dataset.gradY[idx].unsqueeze(0).to(args.device)
    inputs_v = dataset.inputs_target_v[idx].to(args.device).unsqueeze(0)
    inputs_v_ = inputs_v.clone()
    inputs_v_[..., :3] = inputs_v_[..., :3] * scale + torch.from_numpy(shift).cuda()
    verts_ = inputs_v_[..., :3].squeeze()
    if type(dataset.face) == list: # Multiple topology in the same dataset
        face = dataset.face[idx].cpu().numpy()
        dfn_info = dataset.dfn_info_list[idx]
        dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    mesh_expressed = Mesh(verts_.cpu().numpy(), face)

    with torch.no_grad():
        if calc_dfn_info:
            print('Recalculating dfn_info on-the-fly')
            mesh_input = Mesh(verts_.squeeze().cpu().numpy(), face)
            dfn_info = get_dfn_info(mesh_input)
            dfn_info = [_.to(args.device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
        model.update_precomputes(dfn_info)
        if args.img:
            z = model.encode(inputs_v_, img_.to(inputs_v_.device), p2f_t_idxs=p2f_t_idxs, p2f_t_vals=p2f_t_vals, f2v_t_idxs=f2v_t_idxs, f2v_t_vals=f2v_t_vals, N_F=face.shape[0], batch_gradX=gradX, batch_gradY=gradY)
        else:
            z = model.encode(inputs_v_, batch_gradX=gradX, batch_gradY=gradY)
    z[..., :53] += torch.from_numpy(offset).cuda()
    return z, mesh_expressed

def deform_ict(z, face_model, vedo_face_model):
    # Deform the ict face by the ict blendshapes
    z[z > 1] = 1
    z[z < -1] = -1
    deformed_vertices = face_model.deform(z[:, :53])[0] # Select z_FACS
    vedo_face_model.points(deformed_vertices)
    return vedo_face_model

def calc_new_mesh(args,
                normalizer,
                model,
                myfunc,
                mesh,
                z,
                operators,
                dfn_info,
                img=None):
    """
    z: latent code
    
    """
    lu_solver, idxs, vals, rhs = operators
    # calc center of model
    cents = calc_cent(mesh)
    cents = torch.from_numpy(cents).float().unsqueeze(0)

    # normals
    _, norms = calc_norm(mesh)
    norms = torch.from_numpy(norms).float().unsqueeze(0)

    # set inputs
    inputs = torch.cat([cents, norms], dim=-1)
    inputs = inputs.to('cuda')
    norms_v = torch.from_numpy(igl.per_vertex_normals(mesh.vertices, mesh.faces)).float()

    if torch.isnan(norms_v).any():
        # If something wrong with the igl computation
        norms_v, _ = calc_norm(mesh)
        norms_v = torch.from_numpy(norms_v).float()
    # set source vertex
    input_source_v = torch.cat([torch.from_numpy(mesh.vertices), norms_v], dim=-1).float().unsqueeze(0)
    input_source_v = input_source_v.to('cuda')

    # get image feture
    img_feat = model.img_feat(img) 

    # set inputs
    inputs = torch.cat([inputs, img_feat.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)

    # if not args.img_only_mlp:
    input_source_v = torch.cat([input_source_v, img_feat.unsqueeze(1).expand(-1, input_source_v.shape[1], -1)], dim=-1)

    with torch.no_grad():
        model.update_precomputes(dfn_info)
        
        if z.shape[0] > 1:
            z = z.reshape(-1, z.shape[-1]) 
            inputs = inputs.repeat(z.shape[0], 1, 1)
            input_source_v = input_source_v.repeat(z.shape[0], 1, 1)
        
        # decode to get mesh
        g_pred, z_iden = model.decode([inputs.float(), input_source_v.float()], z.float())

        # solve for the mesh
        g_pred  = normalizer.inv_normalize(g_pred)
        g_pred  = reconstruct_jacobians(g_pred, repr='matrix')
        out_pred = myfunc(g_pred, lu_solver, idxs, vals, rhs.shape)
        out_pred = out_pred - out_pred.mean(axis=[0, 1], keepdim=True)

        # get the mesh
        mesh_out = Mesh(out_pred[0].detach().cpu().numpy(), mesh.faces)
        theta, omega, A = calc_jacobian_stat(g_pred[0].detach().cpu().numpy())

    if z_iden is not None:
        z_iden = z_iden.detach().cpu().numpy()
    return mesh_out, theta, omega, A, z_iden

def apply_z_code(
                args, \
                model, \
                dataset, \
                face, \
                idx, \
                dfn_info, \
                recalculate_dfn_info, \
                global_shift_source,
                global_scale_source,
                offset, \
                z,
                orig_mesh, \
                mesh_gt, \
                normalizer, \
                myfunc, \
                mesh, \
                mesh_operators, \
                mesh_dfn_info, \
                img, \
                mesh_vedo \
                ):
    z, mesh_expressed = generate_new_z(args, model, dataset, face, idx, dfn_info, calc_dfn_info=recalculate_dfn_info, shift=global_shift_source, scale=global_scale_source, offset=offset)
    mesh_, theta, omega, A, z_iden = calc_new_mesh(args, normalizer,model, myfunc, mesh, z, mesh_operators, mesh_dfn_info, img=img)
    deform_ict(z)
    points = mesh_.vertices
    mesh_vedo.points(points)
    if orig_mesh is not None:
        mesh_gt.points(mesh_expressed.vertices)

    return  z, orig_mesh, mesh_gt




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


class NormalizerV2(object):
    def __init__(self, std_path, device, zero_mean=True):

        self.gradients_std = np.load(std_path)
        self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
        self.gradients_mean = torch.eye(3).view(-1,).unsqueeze(0).unsqueeze(0).to(device).float()

    def normalize(self, tensor):

        return (tensor - self.gradients_mean.to(tensor.device)) / self.gradients_std.to(tensor.device)

    def inv_normalize(self, tensor):

        return tensor * self.gradients_std.to(tensor.device) + self.gradients_mean.to(tensor.device)


class NormalizerImgV2(Normalizer):
    def __init__(self, std_path, device):
        import warnings
        warnings.filterwarnings('ignore')
        with open(std_path, 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()


def load_state_dict(model, state_dict):
    key = list(state_dict.keys())[0]
    if 'module' in key:
        if type(model) != torch.nn.DataParallel:
            state_dict = {name[7:]:value for name, value in state_dict.items()}
    else:
        if type(model) == torch.nn.DataParallel:
            state_dict = {'module.' + name:value for name, value in state_dict.items()}
    if 'module' in list(state_dict.keys())[0]:
        cnn_enc = {name[12 +7:]:value for name, value in state_dict.items() if 'img_encoder' in name}
        cnn_fc = {name[7+7:]:value for name, value in state_dict.items() if 'img_fc' in name}
        exp_enc = {name[8+7:]:value for name, value in state_dict.items() if 'encoder' in name and 'img' not in name}
        iden_enc = {name[10+7:]:value for name, value in state_dict.items() if 'global_pn' in name}
        mlp = {name[7 + 8:]:value for name, value in state_dict.items() if 'linears' in name}
        linear_out = {name[7 + 11:]:value for name, value in state_dict.items() if 'linear_out' in name}
        gns = {name[7 + 4:]:value for name, value in state_dict.items() if 'gns' in name}
        if model.module.img_encoder is not None:
            model.module.img_encoder.load_state_dict(cnn_enc)
            if model.module.img_enc_type == 'cnn':
                model.module.img_fc.load_state_dict(cnn_fc)
        try:
            model.module.encoder.load_state_dict(exp_enc)
        except RuntimeError: # This is a dfn
            exp_enc = {name[4:]:exp_enc[name] for name in list(exp_enc.keys())[10:]}
            model.module.encoder.dfn.load_state_dict(exp_enc)
        if len(iden_enc.keys()) > 0:
            try:
                model.module.global_pn.load_state_dict(iden_enc)
            except RuntimeError:
                iden_enc = {name[4:]:iden_enc[name] for name in list(iden_enc.keys())[10:]}
                model.module.global_pn.dfn.load_state_dict(iden_enc)
        model.module.linears.load_state_dict(mlp)
        model.module.linear_out.load_state_dict(linear_out)
        model.module.gns.load_state_dict(gns)
    else:
        cnn_enc = {name[12:]:value for name, value in state_dict.items() if 'img_encoder' in name}
        cnn_fc  = {name[7:]:value for name, value in state_dict.items() if 'img_fc' in name}
        exp_enc = {name[8:]:value for name, value in state_dict.items() if 'encoder' in name and 'img' not in name}
        iden_enc = {name[10:]:value for name, value in state_dict.items() if 'global_pn' in name}
        mlp = {name[ 8:]:value for name, value in state_dict.items() if 'linears' in name}
        linear_out = {name[11:]:value for name, value in state_dict.items() if 'linear_out' in name}
        gns = {name[4:]:value for name, value in state_dict.items() if 'gns' in name}
        if model.img_encoder is not None:
            model.img_encoder.load_state_dict(cnn_enc)
            if model.img_enc_type == 'cnn':
                model.img_fc.load_state_dict(cnn_fc)
        try:
            model.encoder.load_state_dict(exp_enc)
        except RuntimeError: # This is a dfn
            exp_enc = {name[4:]:exp_enc[name] for name in list(exp_enc.keys())[10:]}
            model.encoder.dfn.load_state_dict(exp_enc)
        if len(iden_enc.keys()) > 0:
            try:
                model.global_pn.load_state_dict(iden_enc)
            except RuntimeError: # This is a dfn
                iden_enc = {name[4:]:iden_enc[name] for name in list(iden_enc.keys())[10:]}
                model.global_pn.dfn.load_state_dict(iden_enc)
        model.linears.load_state_dict(mlp)
        model.linear_out.load_state_dict(linear_out)
        model.gns.load_state_dict(gns)
    return model


def get_dfn_info(mesh, cache_dir=None, map_location='cuda'):
    verts_list = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.faces).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(verts_list, face_list, k_eig=128, op_cache_dir=cache_dir)

    dfn_info = [mass_list[0], L_list[0], evals_list[0], evecs_list[0], gradX_list[0], gradY_list[0], torch.from_numpy(mesh.faces)]
    dfn_info = [_.to(map_location).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info

class renderer:
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