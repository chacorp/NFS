import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

#import igl
import sys
import pickle

from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f'{abs_path}/mesh_utils']
from einops import einsum

import utils.nfr_utils as nfr_utils
from utils import (
    ICT_face_model, 
    Renderer, 
    calc_norm_torch, 
    calc_cent, 
    get_mesh_operators, 
    get_jacobian_matrix,
    vis_rig,
)
from utils.deformation_transfer import deformation_gradient

from .decoder import *
from .encoder import *
from .NFR import *

class Exp(nn.Module):
    def __init__(self, 
                 opts, 
                 mesh_dfn_info=None, 
                 print_param=True, 
                 is_train=False,
                 design="nfr",
                ):
        super().__init__()
        """
        Args:
            opts (Namespace): config options
        """
        self.opts = opts
        self.device = self.opts.device if opts is not None else 'cpu'
        self.is_train = self.opts.is_train if opts is not None else False
        self.ict_face_only = self.opts.ict_face_only if opts is not None else True
        self.design = self.opts.design if opts is not None else design
        self.scale_exp = self.opts.scale_exp if opts is not None else 1.0
        if self.scale_exp != 1.0:
            print(f"[ using scaled expression code!!: {self.scale_exp} ]")
        
        self.in_key = self.opts.feature_type if opts is not None else 'cents&norms'
        self.out_key = self.opts.dec_type if opts is not None else 'disp'
        
        self.img_feat_dim = self.opts.img_feat_dim if opts is not None else 128
        self.rig_dim = self.opts.rig_dim if opts is not None else 128
        self.id_dim = self.opts.id_dim if opts is not None else 128
        self.seg_dim = self.opts.seg_dim if opts is not None else 20
        
        self.hid_dim = 128
        
        ### mesh autoencoder
        in_shape_dict = {'cents&norms':6, 'cents':3, 'cents&norms&seg':7}
        out_shape_dict = {'vert': 3, 'disp': 3, 'jacob': 9, 'Rs': 10}
        
        ### zero vector
        # self.mesh_id_encoder = BaseDiffusionNetEncoder(
        #     in_shape=in_shape_dict[self.in_key],
        #     pre_computes=mesh_dfn_info,
        #     out_shape=self.id_dim,
        # )
        # self.mesh_exp_encoder = BaseDiffusionNetEncoder(
        #     in_shape=in_shape_dict[self.in_key],
        #     pre_computes=mesh_dfn_info,
        #     out_shape=self.hid_dim,
        # )
        # self.mesh_seg_encoder = NewDiffusionNetEncoder(
        #     in_shape=in_shape_dict[self.in_key],
        #     pre_computes = mesh_dfn_info,
        #     mid_shape=self.hid_dim,
        #     out_shape=self.seg_dim,
        #     outputs_at='vertices',
        # )
        # self.mesh_decoder = BaseDiffusionNetEncoder(
        #     # [pos][norm][face region][mesh ID][mesh exp] ---> [disp]
        #     # in_shape=in_shape_dict[self.in_key]+self.hid_dim+self.hid_dim+self.hid_dim,
        #     in_shape=in_shape_dict[self.in_key]+self.hid_dim+self.hid_dim,
        #     pre_computes=mesh_dfn_info,
        #     out_shape=out_shape_dict[self.out_key],
        # )
        if self.design == 'exp':
            # exp
            self.mesh_decoder = BaseDecoder(
                in_dim=in_shape_dict[self.in_key]+self.hid_dim+self.hid_dim,
                out_shape=out_shape_dict[self.out_key],
            )
        else:
            # exp2
            self.mesh_decoder = BaseDecoder(
                in_dim=in_shape_dict[self.in_key]+self.hid_dim+self.hid_dim,
                out_shape=out_shape_dict[self.out_key],
            )
            
        # print all layer params number
        if print_param:
            self.print_parameter_num()
        
        self.criterion = nn.MSELoss()
        self.calc_norm_torch = calc_norm_torch
        self.calc_cent = calc_cent
        self.get_jacobian_matrix = get_jacobian_matrix
        
        self.renderer = Renderer(view_d=2.5, img_size=256, fragments=True)
        
        self.ict_face_model = ICT_face_model(face_only=False, device=self.device)
        self.ict_face_model_fo = ICT_face_model(face_only=True, device=self.device)
        
        self.ict_neutral = torch.from_numpy(self.ict_face_model.neutral_verts).float().to(self.device)
        self.ict_faces = torch.from_numpy(self.ict_face_model.faces).long().to(self.device)
        
        self.ict_neutral_fo = torch.from_numpy(self.ict_face_model_fo.neutral_verts).float().to(self.device)
        self.ict_faces_fo = torch.from_numpy(self.ict_face_model_fo.faces).long().to(self.device)
        
        #---------------------------------------------------------------------------------
        if self.opts.seg_dim == 20:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot.npy'
        elif self.opts.seg_dim == 24:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_24.npy'
        elif self.opts.seg_dim == 14:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_14.npy'
        elif self.opts.seg_dim == 6:
            seg_npy = f'{abs_path}/utils/ict/ICT_segment_onehot_06.npy'
        else:
            raise NotImplementedError(f"no segment map for seg_dim: {self.opts.seg_dim}")
        
        self.ict_vert_segment = torch.from_numpy(np.load(seg_npy)).to(self.device)
        self.ict_vert_segment = self.ict_vert_segment.argmax(-1).long()
        self.ict_vert_segment_fo = self.ict_vert_segment[:self.ict_face_model_fo.v_num]
        #---------------------------------------------------------------------------------
        
        
        #---------------------------------------------------------------------------------
        self.get_mesh_operators = get_mesh_operators
        if self.opts.dec_type=='jacob':
            self.normalizer = nfr_utils.Normalizer(self.opts.std_file, self.device)
            self.myfunc = deformation_gradient.apply

    def print_parameter_num(self):
        """Prints parameters
        """
        print("===========< EXP >===========")
        print(f"[mesh_decoder]: \t{self.count_parameters(self.mesh_decoder)}")
        print(f"[total]: \t{self.count_parameters(self)}")
        print("===============================") 
        
    def count_parameters(self, model):
        try:
            return sum(p.numel() for p in model if p.requires_grad)
        except:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def set_neutral_ict(self, ict_basedir):
        self.ict_basedir = ict_basedir
        self.ict_precompute = f'{self.ict_basedir}/precompute-synth-fullhead'
        self.ict_precompute_fo = f'{self.ict_basedir}/precompute-synth-face_only'
    
        self.neu_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute, f"100_dfn_info.pkl"), 'rb'))
        neu_operators = os.path.join(self.ict_precompute, f"100_operators.pkl")
        self.neu_operators = pickle.load(open(neu_operators, mode='rb'))
        self.neu_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute, f"100_img.npy"))).to(self.device).float()
        
        self.neu_fo_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute_fo, f"100_dfn_info.pkl"), 'rb'))
        neu_operators = os.path.join(self.ict_precompute_fo, f"100_operators.pkl")
        self.neu_fo_operators = pickle.load(open(neu_operators, mode='rb'))
        self.neu_fo_img = torch.from_numpy(np.load(os.path.join(self.ict_precompute_fo, f"100_img.npy"))).to(self.device).float()
        self.local_feat_ict=None
            
    def get_mesh_decoder_parameters(self):
        return self.mesh_decoder.parameters()
        
    def get_mesh_autoencoder_parameters(self):
        """no audio encoder"""
        param_list= [
            *self.mesh_decoder.parameters(), 
        ]
        return param_list
    
    def get_img_feat(self, img):
        """
        Args:
            img (torch.tensor): rendered image [1,H,W,C] or [H,W,C]
        Returns:
            image feature
        """
        if len(img.shape) < 4:
            img = img[None]
        return self.img_fc(self.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
          
    def get_img_feat_from_mesh(self, tgt_mesh, return_feat=False):
        """
        Args
        --------
            tgt_mesh (trimesh): trimesh.Trimesh
            return_feat (bool): if True, return image feature
        Returns
        --------
            tgt_img (torch.tensor): rendered image
        """
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        if return_feat:
            tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
            return tgt_img_feat
        else:
            return tgt_img
    
    def get_precomputes(self, tgt_mesh):
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        tgt_dfn_info = nfr_utils.get_dfn_info(tgt_mesh, map_location=self.device)
        return tgt_img, tgt_dfn_info
    
    def calc_vert(self, pred_jacob, myfunc, operators):
        """Poisson solve
        Args
        --------
            pred_jacob (torch.tensor): jacobian of the mesh
            myfunc (torch.autograd.Function): deformation_gradient
            operators (tuple): tuple of mesh operators for poisson solving \
                               (SuperLU, face_to_vert_idxs, face_to_vert_values, cupy csr matrix)
        Return
        --------
            out_pred (torch.tensor): deformed vertices of the mesh
        """
        lu_solver, idxs, vals, rhs = operators
        pred_vert = myfunc(pred_jacob, lu_solver, idxs, vals, rhs.shape)
        pred_vert = pred_vert.float() # float64 -> float32
        pred_vert = pred_vert - pred_vert.mean(dim=1, keepdim=True)
        return pred_vert
    
    def L2_regularization(self, pred):
        """
        Args:
            pred (torch.tensor): predicted expression code
        
        Returns:
            loss
        """
        #loss = F.normalize(pred, p=2.0, dim=-1, eps=1e-12)
        loss = LA.norm(pred, dim=-1).mean()
        return loss
    
    def non_ict_loss(self, pred):
        """Reference from Neural Face Rigging for Animating and Retargeting Facial Meshes in the Wild [Qin et al. 2023], Eq.(3)
        L_FACS = {   
             -x, x < 0 
              0, 0 <= x < 1
            x-1, x > 1
        }
        Args:
            pred (torch.tensor): predicted expression code
        
        Returns:
            loss
        """
        
        loss = torch.where(pred < 0, -pred, torch.where(pred > 1, pred - 1, torch.zeros_like(pred))).mean()
        return loss
        
    def label_smoothing(self, one_hot, smoothing=0.2):
        """Reference: function borrrowed from DiffusionNet github -> but it is not used!
        Args:
            one_hot (torch.tensor):
            smoothing (float):
        """
        n_class = one_hot.shape[-1]
        one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        return one_hot
        
    def get_local_feature(self, vertices, faces, img_feat, at='verts'):
        """
        Args
        ----------
            vertices (torch.tensor): [B, V, 3] vertices from the mesh
            faces (torch.tensor): [F, 3] vertex indices of each triangle
            img_feat (torch.tensor): [B, 128] feature from CNN
        
        Returns
        ----------
            local_feat (torch.tensor) [B, V, 6] / [B, F, 6]
        """
        if at=='verts':
            verts_pos = vertices # [1, V, 3]
            verts_nrm = self.calc_norm_torch(verts_pos, faces, at='verts') # [1, V, 3]
            
            B, V, _ = vertices.shape
            verts_img_feat = img_feat.repeat(1, V, 1) # [1, V, 128]
            local_feat = torch.cat([verts_pos, verts_nrm, verts_img_feat], dim=-1) # [1, V, 3+3+128]
        else:
            verts_pos = vertices # [1, V, 3]
            tri_centr = self.calc_cent(verts_pos.squeeze(0), faces, mode='torch').unsqueeze(0)
            tri_norms = self.calc_norm_torch(verts_pos, faces)

            B, F = vertices.shape[0], faces.shape[0]
            tri_img_feat = img_feat.repeat(1, F, 1)
            local_feat = torch.cat([tri_centr, tri_norms, tri_img_feat], dim=-1) # [1, V, 3+3+128]
            
        return local_feat
    
    def get_local_feature_no_img(self, vertices, faces, at='verts'):
        """
        Args
        ----------
            vertices (torch.tensor): [B, V, 3] vertices from the mesh
            faces (torch.tensor): [F, 3] vertex indices of each triangle
        
        Returns
        ----------
            local_feat (torch.tensor) [B, V, 6] / [B, F, 6]
        """
        if at=='verts':
            verts_pos = vertices # [1, V, 3]
            verts_nrm = self.calc_norm_torch(verts_pos, faces, at='verts') # [1, V, 3]
            
            B, V, _ = vertices.shape
            local_feat = torch.cat([verts_pos, verts_nrm], dim=-1) # [1, V, 3+3]
        else:
            verts_pos = vertices # [1, V, 3]
            tri_centr = self.calc_cent(verts_pos, faces, mode='torch')
            tri_norms = self.calc_norm_torch(verts_pos, faces)

            B, F = vertices.shape[0], faces.shape[0]
            local_feat = torch.cat([tri_centr, tri_norms], dim=-1) # [1, V, 3+3]
            
        return local_feat
        
    def encode_id(self, vert_feat, dfn_info):
        """batch size is 1
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            id_code (torch.tensor): [B, ID] identity code per batch
        """
        self.mesh_id_encoder.update_precomputes(dfn_info)
        id_code = self.mesh_id_encoder(vert_feat)
        return id_code
    
    def encode_exp(self, vert_feat, dfn_info, batch_process=True):
        """batch is frame
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            exp_code (torch.tensor): [B, Exp] expression code per batch
        """
        self.mesh_exp_encoder.update_precomputes(dfn_info)
        if batch_process:
            exp_code = self.mesh_exp_encoder(vert_feat) # [1, Rig]
        else:
            exp_code = []
            for v_f in vert_feat:
                exp_c = self.mesh_exp_encoder(v_f[None])
                exp_code.append(exp_c)
            exp_code = torch.vstack(exp_code) #------------- [W, Rig]
        return exp_code
    
    def encode_seg(self, vert_feat, dfn_info):
        """batch size is 1
        Args
        ----------
            vert_feat (torch.tensor): [B, V, 6] pre-vertex features
            dfn_info (list): mesh precompute features for DiffusionNet
        
        Returns
        ----------
            seg_code (torch.tensor): [B, Exp] segment code per batch
        """
        self.mesh_seg_encoder.update_precomputes(dfn_info)
        seg_code = self.mesh_seg_encoder(vert_feat) # [1, ID]
        return seg_code
    
    def get_inputs_ict(self, pred_exp_coeff):
        """get decoder input from ict neutral face mesh
        Args:
            pred_exp_coeff (torch.tensor): not used in the process
        
        Returns:
            inputs_ict (tuple): inputs for decoder
        """
        img_feat_ict = self.get_img_feat(self.neu_img).unsqueeze(1) # [B, 1, 128]
        template_ict = self.ict_neutral[None]
        operators_ict = self.neu_operators
        faces_ict = self.ict_faces
        dfn_info_ict = self.neu_dfn_info
        
        local_feat_ict = self.get_local_feature(template_ict, faces_ict, img_feat_ict)
        B = pred_exp_coeff.shape[0]
        pred_id_coeff_ict = self.encode_id(local_feat_ict, dfn_info_ict).repeat(B,1)# [1, ID]
        if 'new2' in self.design:
            pred_seg_coeff_ict = self.encode_seg(local_feat_ict, dfn_info_ict)# [1, V, Seg]
        else:
            pred_seg_coeff_ict = None
        
        if self.opts.dec_type=='jacob':
            local_feat_ict = self.get_local_feature(template_ict, faces_ict, img_feat_ict, at='faces')
        return (
            local_feat_ict.repeat(B, 1, 1), 
            pred_exp_coeff, 
            pred_id_coeff_ict, 
            pred_seg_coeff_ict.repeat(B, 1, 1), 
            None, # style_emb #(not used)
            template_ict.repeat(B, 1, 1), 
            faces_ict, 
            operators_ict,
        )
    
    @torch.no_grad()
    def encode_mesh(self,
                    gt_vertices,
                    src_mesh, 
                    tgt_mesh, 
                    batch_process=False,
                    use_filter=False,
                    kernel_size=5,
                    sigma=1.2,
                    use_scale=False,
                    scale=1.0,
                ):
        
        ## target mesh local feature
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
        tgt_dfn_info = nfr_utils.get_dfn_info(tgt_mesh, map_location=self.device)
        tgt_verts = torch.from_numpy(tgt_mesh.vertices)[None].to(self.device).float()
        tgt_faces = torch.from_numpy(tgt_mesh.faces).to(self.device)
        vert_feat = self.get_local_feature(tgt_verts, tgt_faces, tgt_img_feat)# [1, V, 3+3+128]
                
        with torch.no_grad():
            pred_id_coeff = self.encode_id(vert_feat, tgt_dfn_info)

            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, tgt_dfn_info)# [1, V, Seg]
            else:
                pred_seg_coeff = None
        
        #pred_id_coeff = pred_id_coeff.unsqueeze(1) # ----------------------- [1, 1, ID]
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ## source mesh local feature
        gt_vertices = gt_vertices.to(self.device)
        src_img = self.renderer.render_img(src_mesh).float().to(self.device)
        src_img_feat = self.get_img_feat(src_img).unsqueeze(1) #------------------- [1, 1, 128]
        src_dfn_info = nfr_utils.get_dfn_info(src_mesh, map_location=self.device)
        src_faces = torch.from_numpy(src_mesh.faces).to(self.device)
        vert_feat_exp = self.get_local_feature(gt_vertices, src_faces, src_img_feat).float()# [W, V, 3+3+128]
        
        with torch.no_grad():
            pred_exp_coeff = self.encode_exp(vert_feat_exp, src_dfn_info, batch_process=batch_process)# [W, Rig]
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff, 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        
        if use_scale:
            pred_exp_coeff = pred_exp_coeff * scale
            
        return pred_exp_coeff, pred_id_coeff, pred_seg_coeff, vert_feat
    
    def encode_mesh_grad(self, vert_feat, vert_feat_exp, dfn_info):
        pred_seg_coeff = None
        pred_id_coeff = self.encode_id(vert_feat, dfn_info) # [1, ID]
        pred_exp_coeff = self.encode_exp(vert_feat_exp, dfn_info) # [W, Rig]
        
        if 'new2' in self.design:
            pred_seg_coeff = self.encode_seg(vert_feat, dfn_info) # [1, V, Seg]
                
        return pred_id_coeff, pred_exp_coeff, pred_seg_coeff
    
    def gaussian_kernel1d(self, kernel_size=5, sigma=1.0):
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def apply_gaussian_filter(self, tensor, kernel_size, sigma):
        """Applies a Gaussian filter to the tensor.
        Args:
            tensor (torch.tensor): input tensor
            kernel_size (int): size of the kernel
            sigma (float): sigma value for gaussian filter
        Returns:
            filtered_tensor
        """
        kernel_size = int(kernel_size)

        # Generate the 1D Gaussian kernel
        kernel = self.gaussian_kernel1d(kernel_size, sigma)
        kernel = kernel.reshape(1, 1, -1).to(self.device) # (out_channels, in_channels, kernel_size)
        kernel = kernel.repeat(tensor.size(1), 1, 1) # [128, 1, kernel_size]
        tensor = tensor.transpose(0, 1).unsqueeze(0) # [1, 128, T]

        filtered_tensor = F.conv1d(tensor, kernel, padding=(kernel_size // 2), groups=tensor.size(1))

        # Transpose back to original shape
        filtered_tensor = filtered_tensor.squeeze(0).transpose(0, 1)
        return filtered_tensor
    
    def decode_grad(self, inputs, batch_process=True):
        """
        Args
        ----------
            inputs (tuple):
                local_feat (torch.tensor): [1, V, 6]
                pred_exp_coeff (torch.tensor): [B, Exp]
                pred_id_coeff (torch.tensor): [B, ID]
                pred_seg_coeff (torch.tensor): [B, V, Seg]
                style_emb (torch.tensor): [B, feat] (not used)
                template (torch.tensor): [B, V, 3] vertices
                faces (torch.tensor): [F, 3] triangle indicies
                operators (tuple): SuperLU, Laplacian mat ...
            batch_process (bool): if True, process in batch
        
        Returns
        ----------
            pred_outputs, pred_jacobians: predicted vertices and jacobians
        """
        (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators) = inputs
        
        B, _ = pred_exp_coeff.shape
        _, V, _  = local_feat.shape
        
        if batch_process:
            #local_feat = local_feat.repeat(W, 1, 1)
            pred_exp_coeff_repeat = pred_exp_coeff.unsqueeze(1).repeat(1, V, 1)
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(1, V, 1)
            
            if 'nfr' in self.design:
                inputs = torch.cat([local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat], dim=-1)
                pred_outputs = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
            else:
                pred_outputs = self.mesh_decoder(local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat, pred_seg_coeff)
        else:
            pred_outputs = []
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(1, V, 1)  #--------------- [1, V, ID]
            
            for pred_rig in pred_exp_coeff.unsqueeze(1):
                if 'nfr' in self.design:
                    single_pred_rig = pred_rig[None].repeat(1, V, 1), #--------------- [1, V, Rig]
                    
                    inputs = torch.cat([local_feat, single_pred_rig, pred_id_coeff_repeat], dim=-1).float() #-- [1, V, 134+Rig+ID]
                    tmp_vertices = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
                else:
                    tmp_vertices = self.mesh_decoder(local_feat, pred_rig[None], pred_id_coeff_repeat, pred_seg_coeff)
                    
                pred_outputs.append(tmp_vertices)
            pred_outputs = torch.vstack(pred_outputs)  #-------------------- [W, V, 3]
            
        if self.opts.dec_type=='jacob':
            pred_jacobians = self.normalizer.inv_normalize(pred_outputs)
            pred_jacobians = nfr_utils.reconstruct_jacobians(pred_jacobians, repr='matrix')
            
            pred_outputs = self.calc_vert(pred_jacobians, self.myfunc, operators)
        else:
            if self.opts.dec_type=='disp':
                pred_outputs = template + pred_outputs
            pred_jacobians = self.get_jacobian_matrix(pred_outputs, faces, template, return_torch=True)
            
        return pred_outputs, pred_jacobians
    
    def decode_grad_no_seg(self, inputs):
        """
        Args
        ----------
            inputs (tuple):
                local_feat (torch.tensor): [1, V, 6]
                pred_exp_coeff (torch.tensor): [B, Exp]
                pred_id_coeff (torch.tensor): [B, ID]
                style_emb (torch.tensor): [B, feat] (not used)
                template (torch.tensor): [B, V, 3] vertices
                normals (torch.tensor): [B, V, 3] vertex normals
                faces (torch.tensor): [F, 3] triangle indicies
                operators (tuple): SuperLU, Laplacian mat ...
            batch_process (bool): if True, process in batch
        
        Returns
        ----------
            pred_outputs: predicted vertices
        """
        (local_feat, pred_exp_coeff, pred_id_coeff, template, normals, faces, operators) = inputs
        
        B, _ = pred_exp_coeff.shape
        _, V, _  = local_feat.shape
        
        #local_feat = local_feat.repeat(W, 1, 1)
        pred_exp_coeff_repeat = pred_exp_coeff.unsqueeze(1).repeat(1, V, 1)
        pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(1, V, 1)
        
        inputs = torch.cat([local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat], dim=-1)
        pred_outputs = self.mesh_decoder(inputs)
        
        if self.opts.dec_type == 'jacob':
            pred_jacobians = self.normalizer.inv_normalize(pred_outputs)
            pred_jacobians = nfr_utils.reconstruct_jacobians(pred_jacobians, repr='matrix')
            
            pred_verts = self.calc_vert(pred_jacobians, self.myfunc, operators)
        else:
            if self.opts.dec_type=='disp':
                pred_verts = template + pred_outputs
                # pred_jacobians = self.get_jacobian_matrix(pred_outputs, faces, template, return_torch=True)
                
            elif self.opts.dec_type=='Rs':
                pred_rot= pred_outputs[...,:-1].reshape(B, V, 3, 3)
                pred_s = pred_outputs[...,-1].unsqueeze(-1)
                
                pred_verts = template + einsum(pred_rot.permute(0,1,3,2), normals, 'b v n c, b v c -> b v n') * pred_s
            
        if self.opts.dec_type=='Rs':
            return pred_verts, pred_outputs #, pred_jacobians
        
        return pred_verts #, pred_jacobians
    
    @torch.no_grad()
    def decode(self, 
               inputs, 
               tgt_mesh=None, # required if dec_type=='jacob'
               scale_exp=1.0, 
               return_exp=False,
               batch_process=False,
               use_filter=False,
               kernel_size=5,
               scale_mask=torch.ones(53),
               sigma=1.0
              ):
        """
        ## Note: batch_size is always 1 | used at inference
        Args
        ----------
            inputs:
                local_feat (torch.tensor): [1, V, 6]
                pred_exp_coeff (torch.tensor): [B, 1, Exp]
                pred_id_coeff (torch.tensor): [B, 1, ID]
                pred_seg_coeff (torch.tensor): [B, V, Seg]
                style_emb (torch.tensor): [B, feat]
                template (torch.tensor): [V, 3] vertices
                faces (torch.tensor): [F, 3] triangle indicies
                operators
            tgt_mesh (trimesh.Trimesh):
            scale_exp (float): scale the expression code
            return_exp (bool): if True, return epxression code
        Returns
        ----------
            pred_outputs: [batch_size, W, V, 3]
        """
        (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators) = inputs
        
        if scale_exp != 1.0:
            pred_exp_coeff = pred_exp_coeff * scale_exp
        
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff, 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        W, _ = pred_exp_coeff.shape
        V = local_feat.shape[1]
        
        if batch_process:
            local_feat = local_feat.repeat(W, 1, 1)
            pred_exp_coeff_repeat = pred_exp_coeff.unsqueeze(1).repeat(1, V, 1)
            pred_id_coeff_repeat = pred_id_coeff.unsqueeze(1).repeat(W, V, 1)
            
            if 'nfr' in self.design:
                inputs = torch.cat([local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat], dim=-1)
                pred_outputs = self.mesh_decoder(inputs) # [B, V or VF, out_dim]
                
            else:
                pred_outputs = self.mesh_decoder(local_feat, pred_exp_coeff_repeat, pred_id_coeff_repeat, pred_seg_coeff)
        else:
            pred_outputs = []
            pred_id_coeff_repeat = pred_id_coeff.repeat(1, V, 1) #--------------- [1, V, ID]
        
            for pred_rig in pred_exp_coeff: # pred_rig -> [Rig]
                
                pred_rig = pred_rig[None]
                if 'nfr' in self.design:
                    pred_rig = pred_rig.repeat(1, V, 1) #--------------- [1, V, Rig]
                    input_frame = torch.cat([local_feat, pred_rig, pred_id_coeff_repeat], dim=-1).float() # [1, V, 134+Rig+ID]
                    with torch.no_grad():
                        tmp_vertices = self.mesh_decoder(input_frame) # [B, V or VF, out_dim]
                else:
                    with torch.no_grad():
                        tmp_vertices = self.mesh_decoder(local_feat, pred_rig, pred_id_coeff_repeat, pred_seg_coeff)
                            
                pred_outputs.append(tmp_vertices)
            pred_outputs = torch.vstack(pred_outputs)  #-------------------- [W, V, 3]
            ## empty_cache
            torch.cuda.empty_cache()
        
        
        ## converting outputs
        ## NFR design requires Poisson solving to obtain final vertices
        if self.opts.dec_type=='jacob':
            if operators is None:
                operators = self.get_mesh_operators(tgt_mesh)
            
            pred_jacobians = self.normalizer.inv_normalize(pred_outputs)
            pred_jacobians = nfr_utils.reconstruct_jacobians(pred_jacobians, repr='matrix')
            
            with torch.no_grad():
                pred_outputs = self.calc_vert(pred_jacobians, self.myfunc, operators)
        else:
            if self.opts.dec_type=='disp':
                pred_outputs = template + pred_outputs
            pred_jacobians = self.get_jacobian_matrix(pred_outputs, faces, template, return_torch=True)
            
        if return_exp:
            return pred_outputs, pred_jacobians, pred_exp_coeff
        else:
            return pred_outputs, pred_jacobians
        
            
    def forward(self, batch, teacher_forcing=True, return_all=False, stage=1, epoch=0):
        if stage == 1:
            ### train with ICT only + train mesh autoencoder only (learn rig space)
            return self.stage1_forward(batch, teacher_forcing, return_all, epoch)
        elif stage == 11:
            ### train mesh decoder only
            return self.stage11_forward(batch, teacher_forcing, return_all, epoch)
        else:
            return NotImplementedError
    
    def stage1_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                audio_feat (torch.Tensor):    [B, W, 768] (not used !!!)
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
                img (torch.Tensor):           [B, H, W, C] rendered mesh image
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data]
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff
        gt_rig_params = batch.gt_rig_params
        template = batch.template
        gt_vertices = batch.vertices
        faces = batch.faces
        img = batch.img
        gt_normals = batch.normals
        dfn_info  = pickle.load(open(batch.dfn_info, 'rb'))
        operators = pickle.load(open(batch.operators, mode='rb'))
        ##------------------------------------------------------------------------------
        
        ## Encoding --------------------------------------------------------------------
        ## source & target mesh image feature
        img_feat = self.get_img_feat(img).unsqueeze(1) # [1, 1, 128]
        # source expression face
        vert_feat_exp = self.get_local_feature(gt_vertices, faces, img_feat)  # [W, V, 6]
        # target neutral face
        vert_feat = self.get_local_feature(template, faces, img_feat) # [1, V, 6]
        
        pred_id_coeff, pred_exp_coeff, pred_seg_coeff = self.encode_mesh_grad(vert_feat, vert_feat_exp, dfn_info)
        ##------------------------------------------------------------------------------
        
        
        ## Decoding --------------------------------------------------------------------
        # target local features
        if self.opts.dec_type=='jacob':
            faces_feat = self.get_local_feature(template, faces, img_feat, at='faces')
            local_feat = faces_feat # [1, V, 3+3+128]
        else: # dec_type=='disp'
            local_feat = vert_feat # [1, V, 3+3+128]
        
        style_emb = None # -> not used!
        # target inputs
        if epoch < 100 and mesh_data == 'ict' and self.opts.warmup:
            inputs = (local_feat, gt_rig_params, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators)
        else:
            inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators)
        # inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators)
        pred_outputs, pred_jacobians = self.decode_grad(inputs)
        ##------------------------------------------------------------------------------
        
        
        ## Loss function ---------------------------------------------------------------
        loss = {}
            
        loss["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
        
        pred_normals = self.calc_norm_torch(pred_outputs, faces, at='verts')
        loss["norm_vDec"] = self.criterion(gt_normals, pred_normals)
        
        gt_jacobians = self.get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        loss["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)
        
        if mesh_data == 'ict':
            loss["vert_rIEnc"] = self.criterion(gt_id_coeff, pred_id_coeff)
            loss["vert_rEEnc"] = self.criterion(gt_rig_params, pred_exp_coeff)
            
            if template.shape[1] > 9409:
                ict_model = self.ict_face_model # full_head
                ict_seg = self.ict_vert_segment
            else:
                ict_model = self.ict_face_model_fo # face_only
                ict_seg = self.ict_vert_segment_fo
                
            if not self.opts.no_BP:
                pred_ICT_recon = ict_model.apply_coeffs_batch_torch(pred_id_coeff[:, :100], pred_exp_coeff[:, :53])
                loss["vert_vICT"] = self.criterion(gt_vertices, pred_ICT_recon)
        
            if not self.opts.no_BR:
                with torch.no_grad(): ## no need to pass it to encoders
                    inputs_ict = self.get_inputs_ict(pred_exp_coeff.detach())
                    gt_recon_ICT = ict_model.apply_coeffs_batch_torch(inputs_ict[2][:, :100], pred_exp_coeff[:, :53].detach())
                pred_outputs_ict, _ = self.decode_grad(inputs_ict)
                loss["vert_vICT"] = self.criterion(gt_recon_ICT, pred_outputs_ict[:,:ict_model.v_num])

            if 'new2' in self.design:
                pred_seg_log = F.log_softmax(pred_seg_coeff, dim=-1) ## [V, Seg]
                loss["nll_vSeg"] = 0
                for pred_s_l in pred_seg_log:
                    loss["nll_vSeg"] += F.nll_loss(pred_s_l, ict_seg)
                loss["nll_vSeg"] = loss["nll_vSeg"] / pred_seg_log.shape[0]
            # import pdb;pdb.set_trace()
        else:
            loss["vert_rIEnc"] = self.non_ict_loss(pred_id_coeff[:, :100]) + self.L2_regularization(pred_id_coeff[:, 100:])
            loss["vert_rEEnc"] = self.non_ict_loss(pred_exp_coeff[:, :53]) + self.L2_regularization(pred_exp_coeff[:, 53:])
            
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
            else:
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    def stage11_forward(self, batch, teacher_forcing=True, return_all=False, epoch=0):
        """
        Args
        ----------
            batch (tuple):
                audio_feat (torch.Tensor):    [B, W, 768] (not used !!!)
                id_coeff (torch.Tensor):      [B, 128]    ICT-face model id coeff
                gt_rig_params (torch.Tensor): [B, 128] ICT-face model exp_coeff 
                template (torch.Tensor):      [B, V, 3] mesh vertices
                dfn_info (list(torch.Tensor)): DiffusionNet precomputes (mass, L, eval, evecs, gradX, gradY, faces)
                operators (list(torch.Tensor)): mesh operators for Poisson solve (NFR)
                vertices (torch.Tensor):      [B, V, 3] sequence of mesh vertices (animation)
                faces (torch.Tensor):         [F, 3] mesh faces indices
                img (torch.Tensor):           [B, H, W, C] rendered mesh image
            teacher_forcing (bool): used for Transformer model (not used !!!)
            return_all (bool): if True, return loss and all predicted outputs
            
        Returns
        ----------
            pred_outputs (torch.Tensor): [B, W, V, 3]
        """
        
        #_, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, mesh_data = batch
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data]
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = torch.zeros_like(batch.id_coeff).to(device=self.device, dtype=batch.id_coeff.dtype)
        gt_rig_params = batch.gt_rig_params
        template = batch.template
        gt_vertices = batch.vertices
        faces = batch.faces
        # img = batch.img
        gt_normals = batch.normals
        
        # dfn_info = pickle.load(open(batch.dfn_info, 'rb'))
        if self.opts.dec_type == 'jacob':
            operators = pickle.load(open(batch.operators, mode='rb'))
        else:
            operators = None
        ##------------------------------------------------------------------------------
                
        ## Decoding --------------------------------------------------------------------
        B, V, _ = gt_vertices.shape
        # target local features
        local_feat = self.get_local_feature_no_img(template, faces, at='faces' if self.opts.dec_type=='jacob' else 'verts') # [1, V, 3+3+128]
        
        inputs = (local_feat, gt_rig_params, gt_id_coeff, template, gt_normals, faces, operators)
        if self.opts.dec_type=='Rs':
            pred_verts, pred_outputs = self.decode_grad_no_seg(inputs)
        elif self.opts.dec_type=='jacob':
            pred_verts, pred_outputs = self.decode_grad_no_seg(inputs)
            pred_verts = pred_verts - pred_verts.mean(dim=1, keepdim=True) + gt_vertices.mean(dim=1, keepdim=True)
        else:
            pred_verts = self.decode_grad_no_seg(inputs)
        ##------------------------------------------------------------------------------
        
        
        ## Loss function ---------------------------------------------------------------
        loss = {}
        loss["recon_vDec"] = self.criterion(gt_vertices, pred_verts)
        
        if self.opts.dec_type=='Rs':
            pred_rot= pred_outputs[...,:-1].reshape(B, V, 3, 3)
            pred_I = einsum(pred_rot.permute(0,1,3,2), pred_rot, 'b v n c, b v c m -> b v n m')
            I_mat = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B,V,1,1).to(device=pred_I.device, dtype=pred_I.dtype)
            loss["vert_rot"] = self.criterion(I_mat, pred_I)
        
        # pred_normals = self.calc_norm_torch(pred_outputs, faces, at='verts')
        # loss["norm_vDec"] = self.criterion(gt_normals, pred_normals)
        
        # gt_jacobians = self.get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        # loss["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)
        
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_verts, None, None, None, None
            else:
                return loss, pred_verts, None, None, None, None
        return loss
    
    @torch.no_grad()
    def evaluate(self, batch, batch_process=True, return_all=False, stage=1, epoch=0, newid=None, mode=''):

        ### train with ICT only + train mesh autoencoder only (learn rig space)
        if mode == 'invrig':
            return self.stage1_invrig(batch, newid, batch_process, return_all, epoch)
        else:
            return self.stage1_evaluate(batch, batch_process, return_all, epoch)
    
    def stage1_evaluate(self, batch, batch_process=False, return_all=False, epoch=0):
        """
        ## Note: B (batch_size) is always 1
        Args:
            batch: (tuple)
                audio_feat:    [B, W, 768] audio feature from wav2vec 2.0 -> not used here
                id_coeff:      [1, 128]    ICT-face model id coeff
                gt_rig_params: [B, W, 128] ICT-face model exp_coeff 
                template:      [B, V, 3] mesh vertices
                dfn_info (list): DiffusionNet information
                operators (list): mesh operators
                vertices:      [B, W, V, 3] sequence of mesh vertices (animation)
                faces:         [B, F, 3] mesh faces indices
                img:           [B, 256, 256, 3] rendered mesh image
            batch_process: (bool) if True, process in batch
            return_all: (bool) if True, return loss and all predicted outputs
        Return:
            pred_outputs: (torch.tensor) [B, W, V, 3]
        """
        mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
        
        ## Send to device --------------------------------------------------------------
        # gt_id_coeff = batch.id_coeff.to(self.device)
        # gt_rig_params = batch.gt_rig_params.to(self.device)
        template = batch.template.to(self.device)
        gt_vertices = batch.vertices.to(self.device)
        faces = batch.faces.to(self.device)
        img = batch.img.to(self.device)
        #gt_normals = batch.normals.squeeze()
        operators = pickle.load(open(batch.operators, mode='rb'))
        dfn_info = pickle.load(open(batch.get_dfn_info, mode='rb'))
        ##------------------------------------------------------------------------------
        
        
        ## Encoding --------------------------------------------------------------------
        ## source & target mesh local feature
        img_feat = self.get_img_feat(img).unsqueeze(1) #--------------------------------------- [1, 1, 128]
        vert_feat = self.get_local_feature(template, faces, img_feat)#------------------ [1, V, 3+3+128]
        vert_feat_exp = self.get_local_feature(gt_vertices, faces, img_feat)#----------- [W, V, 3+3+128]
        
        with torch.no_grad():
            pred_id_coeff = self.encode_id(vert_feat, dfn_info)#---------------- [1, ID]
            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, dfn_info)# [1, V, Seg]
            else:
                pred_seg_coeff = None
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ##------------------------------------------------------------------------------
        if self.scale_exp != 1:
            pred_exp_coeff = pred_exp_coeff * self.scale_exp
        ##------------------------------------------------------------------------------
        
        ## Decoding --------------------------------------------------------------------
        with torch.no_grad():
            if self.opts.dec_type=='jacob':
                faces_feat = self.get_local_feature(template, faces, img_feat, at='faces')
                local_feat = faces_feat #----------------------------------------------- [1, V, 3+3+128]
            else:
                local_feat = vert_feat #------------------------------------------------ [1, V, 3+3+128]
        
            #operators = pickle.load(open(operators[0], mode='rb'))
            style_emb = None
            inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, template, faces, operators)
            pred_outputs, _ = self.decode(inputs, batch_process=False)
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        loss = {}
        if self.opts.dec_type=='jacob':
            gt_vertices_zm = gt_vertices - gt_vertices[:, 0:1].mean(dim=1, keepdim=True) # because no translation included
            loss["recon_vDec"] = self.criterion(gt_vertices_zm, pred_outputs)
        else:
            loss["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
            
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
            else:
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    def stage1_invrig(self, batch, newid, batch_process=False, return_all=False, epoch=0):
        """
        ## Note: B (batch_size) is always 1
        Args:
            batch (tuple):
                audio_feat:    [B, W, 768] audio feature from wav2vec 2.0 -> not used here
                id_coeff:      [1, 128]    ICT-face model id coeff
                gt_rig_params: [B, W, 128] ICT-face model exp_coeff 
                template:      [B, V, 3] mesh vertices
                dfn_info (list): DiffusionNet information
                operators (list): mesh operators
                vertices:      [B, W, V, 3] sequence of mesh vertices (animation)
                faces:         [B, F, 3] mesh faces indices
                img:           [B, 256, 256, 3] rendered mesh image
            newid (tuple):
                tgt_coeff:     [1, 128]    ICT-face model id coeff
                tgt_idx: (int)
            batch_process: (bool) if True, process in batch
            return_all: (bool) if True, return loss and all predicted outputs
        Return:
            pred_outputs: (torch.tensor) [B, W, V, 3]
        """
        (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data = batch
        B, W, _ = audio_feat.shape
        V = template.shape[1]
        
        ## Send to device --------------------------------------------------------------
        #gt_id_coeff = id_coeff.to(self.device).float()
        gt_rig_params = gt_rig_params.to(self.device).float().squeeze(0)
        template = template.to(self.device).float()
        gt_vertices = vertices.to(self.device).float().squeeze(0)
        faces = faces.to(self.device).squeeze(0)
        img = img.to(self.device).float()
        
        
        # get New target mesh (neutral face and animated)
        (tgt_coeff, tgt_idx) = newid
        #p_mode = 'face_only' if self.ict_face_only else 'fullhead'
        tgt_img = np.load(os.path.join(self.ict_precompute, f"{tgt_idx:03d}_img.npy")) # ----- [1, 256, 256, 3]
        tgt_img = torch.from_numpy(tgt_img).to(self.device).float()
        tgt_coeff = torch.from_numpy(tgt_coeff).to(self.device).float()
        tgt_gt_rig = gt_rig_params[:,:53].to(self.device).float()
        tgt_dfn_info  = pickle.load(open(os.path.join(self.ict_precompute, f"{tgt_idx:03d}_dfn_info.pkl"), 'rb')) # list[ ... ]
        tgt_operators = os.path.join(self.ict_precompute, f"{tgt_idx:03d}_operators.pkl") # list[ ... ]
        tgt_operators = pickle.load(open(tgt_operators, mode='rb'))
        
        # displacements
        tgt_id_disps = torch.einsum('k,kls->ls', tgt_coeff, self.ict_face_model.id_basis)[:self.ict_face_model.v_num]
        tgt_exp_disp = torch.einsum('jk,kls->jls', tgt_gt_rig, self.ict_face_model.exp_basis)[:,:self.ict_face_model.v_num]
        
        tgt_template  = self.ict_neutral.to(self.device) + tgt_id_disps # neutral face
        tgt_gt_vertices = tgt_template + tgt_exp_disp # facial expressions
        
        # send to device
        tgt_template = tgt_template[None].to(self.device).float()
        tgt_gt_vertices = tgt_gt_vertices.to(self.device).float()
        ##------------------------------------------------------------------------------
        
        
        
        
        ## Encoding --------------------------------------------------------------------
        ## target mesh feature
        tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------------------- [1, 1, 128]
        vert_feat = self.get_local_feature(tgt_template, faces, tgt_img_feat)#---------- [1, V, 3+3+128]
        
        ## source mesh feature
        img_feat = self.get_img_feat(img).unsqueeze(1) #--------------------------------------- [1, 1, 128]
        vert_feat_exp = self.get_local_feature(gt_vertices, faces, img_feat)#----------- [W, V, 3+3+128]

        with torch.no_grad():
            # target id_code
            pred_id_coeff = self.encode_id(vert_feat, tgt_dfn_info)#-------------------- [1, ID]

            # target seg_code
            if 'new2' in self.design:
                pred_seg_coeff = self.encode_seg(vert_feat, tgt_dfn_info)# [1, V, Seg]

            # source exp_code
            pred_exp_coeff = self.encode_exp(vert_feat_exp, dfn_info, batch_process=batch_process)# [W, Rig]
        torch.cuda.empty_cache()
        
                
        ## Decoding --------------------------------------------------------------------
        with torch.no_grad():
            if self.opts.dec_type=='jacob':
                faces_feat = self.get_local_feature(tgt_template, faces, tgt_img_feat, at='faces')
                local_feat = faces_feat #--------------------------------------------------- [1, V, 3+3+128]
            else:
                local_feat = vert_feat #---------------------------------------------------- [1, V, 3+3+128]
            
                operators = pickle.load(open(operators[0], mode='rb'))
                style_emb = None
                inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_template, faces, tgt_operators)
                pred_outputs, pred_jacobians = self.decode(inputs)
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
            
        loss = {}
        if self.opts.dec_type=='jacob':
            loss["recon_vDec"] = self.criterion(tgt_gt_vertices, pred_outputs)
        else:
            loss["recon_vDec"] = self.criterion(tgt_gt_vertices, pred_outputs)
            
        if return_all:
            if self.design == 'nfr': 
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, None
            else:
                return loss, pred_outputs, None, pred_exp_coeff, pred_id_coeff, pred_seg_coeff
        return loss
    
    
    @torch.no_grad()
    def inference(self, 
                     gt_vertices, 
                     src_mesh, 
                     tgt_mesh, 
                     batch_process=False, 
                     return_all=False,
                     return_exp=False, 
                     use_filter=False,
                     use_scale=False,
                     scale_mask=torch.ones(53)*2.0,
                     kernel_size=3,
                     sigma=1.0
                    ):
        """
        ## Note: batch_size is always 1
        Args:
            gt_vertices (torch.tensor): mesh animation vertices
            src_mesh (trimesh): source triangle mesh
            tgt_mesh (trimesh): target triangle mesh
        Return:
            pred_outputs: [batch_size, W, V, 3]
        """
        ## get data and send to device
        if len(gt_vertices.shape) < 3:
            gt_vertices = gt_vertices.reshape(gt_vertices.shape[0], -1, 3)
            
        pred_exp_coeff, pred_id_coeff, pred_seg_coeff, vert_feat = self.encode_mesh(
            gt_vertices,
            src_mesh, 
            tgt_mesh, 
            batch_process=batch_process
        )
        ## empty_cache
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        
        ##------------------------------------------------------------------------------
        if self.scale_exp != 1.0:
            pred_exp_coeff = pred_exp_coeff * self.scale_exp
                    
        if use_scale:
            pred_exp_coeff[:,:53] = pred_exp_coeff[:,:53] * scale_mask[None].to(device=pred_exp_coeff.device)
            
        if use_filter:
            pred_exp_coeff = self.apply_gaussian_filter(
                pred_exp_coeff.squeeze(1), 
                kernel_size=kernel_size, 
                sigma=sigma
            )
        ##------------------------------------------------------------------------------
        
        if self.design == 'nfr':
            pred_exp_coeff = pred_exp_coeff.squeeze()
        
        ## Decode ----------------------------------------------------------------------
        tgt_verts = torch.from_numpy(tgt_mesh.vertices)[None].to(self.device)
        tgt_faces = torch.from_numpy(tgt_mesh.faces).to(self.device)
        tgt_operators = self.get_mesh_operators(tgt_mesh)
        
        if self.opts.dec_type=='jacob':
            tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
            tgt_img_feat = self.get_img_feat(tgt_img).unsqueeze(1) #------------------- [1, 1, 128]
            faces_feat = self.get_local_feature(tgt_verts, tgt_faces, tgt_img_feat, at='faces')
            local_feat = faces_feat # [1, F, 3+3+128]
        else:
            local_feat = vert_feat # [1, V, 3+3+128]

        style_emb = None
        inputs = (local_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_verts, tgt_faces, tgt_operators)
            
        with torch.no_grad():
            pred_outputs, pred_jacobians = self.decode(
                inputs,
                batch_process=False,
            )
            
        ## empty_cache
        torch.cuda.empty_cache()
        ##------------------------------------------------------------------------------
        
        if return_all:
            return pred_outputs, (vert_feat, pred_exp_coeff, pred_id_coeff, pred_seg_coeff, style_emb, tgt_verts, tgt_faces, tgt_operators)
        if return_exp:
            return pred_outputs, pred_exp_coeff
        return pred_outputs