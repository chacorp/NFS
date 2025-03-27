import os
import glob
import json
import yaml
import pickle
import random
import numpy as np
import argparse

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[0].absolute())
sys.path+=[abs_path]
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import trimesh
import matplotlib.pyplot as plt

from models import NFS

import utils.nfr_utils as nfr_utils
from utils.arg_util import YamlArgParser
from dataloader_mesh import (
    NFSDataset,
    InvRigDataset,
)

from utils.ckpt_utils import *
from utils import (
    ICT_face_model, 
    plot_image_array, 
    calc_cent,
    get_mesh_operators,
    get_jacobian_matrix,
    Renderer,
    render_w_audio,
    render_wo_audio,
    calc_norm_torch,
    vis_rig,
)

from utils.deformation_transfer import deformation_gradient

def Options():
    parser = argparse.ArgumentParser(description='NFS evaluation')
    parser.add_argument('-c', '--config', default='config/train.yml', help='config file path')
    parser.add_argument("--feat_dim",     type=int,   default=128,    help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--rig_dim",      type=int,   default=128,    help='rig dim')
    parser.add_argument("--seg_dim",      type=int,   default=20,    help='rig dim')
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--fps",          type=int,   default=30)
    
    parser.add_argument("--audio_type",   type=str,   default="wav2vec2", help="wav2vec2 or hubert")
    parser.add_argument("--feat_level",   type=str,   default="05",   help="wav2vec or hubert")
    parser.add_argument("--input_dim",    type=int,   default=768,    help='1024 for hubert; 768 for wav2vec2; 21 for logits features')

    parser.add_argument("--tb",           action='store_true')
    parser.add_argument("--log_dir",      type=str,   default="ckpt")
    parser.add_argument("--max_epoch",    type=int,   default=500,    help='number of epochs')
    parser.add_argument("--lambda_recon", type=float, default=1.0,    help='recon lambda for encoder')
    parser.add_argument("--lambda_temp",  type=float, default=0.0,    help='temp lambda for encoder')
    parser.add_argument("--lr",           type=float, default=0.0001, help='learning rate')
    
    parser.add_argument("--window_size",  type=int,   default=8,      help='window size')

    parser.add_argument("--data_dir",     type=str,   default="/data/ICT-audio2face/data_30fps")
    parser.add_argument("--seed",         type=int,   default=1234,   help='random seed')
    parser.add_argument("--ckpt",         type=str,   default=None)
    
    parser.add_argument("--recon_type",   type=str,   default="baseline", help="baseline, distill, hybrid")
    parser.add_argument("--dec_type",     type=str,   default="disp", help="vert, disp, jacob")
    parser.add_argument("--design",       type=str,   default="nfr",  help="nfr, new, new2, nfr-adain, codetalker")
    
    parser.add_argument("--ict_face_only",action='store_true', help="if True, use face region only")
    
    ## rigformer experimental feature
    parser.add_argument("--learn_rig_emb",action='store_true', help="start token")
    
    ## training stages
    parser.add_argument("--mesh_d",       action='store_true', help="train mesh decoder")
    parser.add_argument("--stage1",       action='store_true', help="test stage1")
    parser.add_argument("--stage11",      action='store_true', help="train stage11")
    parser.add_argument("--segment",      action='store_true', help="test segment")
    
    parser.add_argument("--debug",        action='store_true')
    
    parser.add_argument("--selection",    type=int, default=20, help='dataset selection')
    parser.add_argument("--NFR",          action='store_true')
    parser.add_argument("--on_ict",       action='store_true')
    parser.add_argument("--inv_rig",      action='store_true')
    
    parser.add_argument("--use_decimate", dest='use_decimate', action='store_true')
    parser.set_defaults(use_decimate=False)

    parser.add_argument("--render",       action='store_true', help="render")
    parser.add_argument("--save_vert",    action='store_true', help="render")
    
    parser.add_argument("--scale_exp",    type=float, default=1.0, help='temp lambda for encoder')
    parser.set_defaults(is_train=True)
    
    
    args = parser.parse_args()
    return args
        
class NFR_helper():
    def __init__(self, opts=None, device='cpu'):
        
        self.offset = np.zeros((1, 53))
        self.scale = 1.0
        self.shift = np.array([0, 0, 0])
        self.latent_dim = 128
        self.device = device
        
        self.criterion = nn.MSELoss()
        
        # mesh normalizer
        self.normalizer = nfr_utils.Normalizer(f'{abs_path}/data/ICT_live_100', 'cuda:0')

        # image feature normalizer
        self.img_normalizer = nfr_utils.Normalizer_img(f'{abs_path}/data/MF_all_v5', 'cuda:0')

        # set pytorch3d renderer
        #self.renderer = myutils.renderer(view_d=2.5, img_size=256, fragments=True)
        self.renderer = Renderer(view_d=2.5, img_size=256, fragments=True)
        
        # Model loading
        self.myfunc = deformation_gradient.apply
        
        self.ict_face_only = opts.ict_face_only if opts is not None else True
        self.ict_face_model = ICT_face_model(face_only=self.ict_face_only, device=self.device)
        self.ict_neutral = self.ict_face_model.neutral_verts
        self.ict_neutral = torch.from_numpy(self.ict_neutral).to(self.device)
        
        self.get_mesh_operators = get_mesh_operators
        
        import pickle
        ## dummy
        dfn_info = pickle.load(open(f'{abs_path}/utils/m00_dfn_info.pkl', 'rb')) # list[ ... ]
        print(f"Loading... pretrained NFR")
        
        self.model = self.model_loading(None, dfn_info)

    def model_loading(self, args, dfn_info):
        global_encoder_in_shape = 6 #if args.feature_type == 'cents&norms' else 12
        in_shape = 6
        from models import latent_space
        model = latent_space(global_encoder_in_shape,
                            in_shape=in_shape,
                            out_shape=9,
                            pre_computes=dfn_info, 
                            latent_shape=128, 
                            iden_blocks=2, 
                            hid_shape=256,  
                            residual=False, 
                            global_pn=True, 
                            sampling=0, 
                            number_gn=32, 
                            dfn_blocks=4, 
                            global_pn_shape=100, 
                            img_encoder='cnn',  
                            img_feat=128, 
                            img_only_mlp=False, 
                            img_warp=False)
        ckpt = torch.load(
            f'{abs_path}/experiments/ICT_augment_cnn_ext_dfn4_grad/ICT_augment_cnn_ext_dfn4_grad_0.pth', 
            map_location='cuda:0'
        )
        # ckpt.keys() == dict_keys(['epoch', 'model', 'optim', 'lr_sched', 'args'])
        model = nfr_utils.load_state_dict(model, ckpt['model'])

        if model.global_pn is not None:
            model.global_pn.update_precomputes(dfn_info)
        model.float()
        model.to('cuda')
        return model

    def get_img_feat(self, img):
        if len(img.shape) < 4:
            img = img[None]
        return self.model.img_fc(self.model.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
    
    def get_inputs(self, vertices, faces, at='verts'):
        """
        Args:
            vertices (torch.tensor): [B, V, 3] vertices from the mesh
            faces (torch.tensor): [F, 3] vertex indices of each triangle
        Return:
            inputs_v (torch.tensor): [B, F, 6]
        """
        if at=='verts':
            verts_pos = vertices # [1, V, 3]
            verts_nrm = calc_norm_torch(verts_pos, faces, at='verts') # [1, V, 3]
            
            inputs_v = torch.cat([verts_pos, verts_nrm], dim=-1) # [1, V, 3+3]
        else:
            verts_pos = vertices # [1, V, 3]
            tri_cnt = calc_cent(verts_pos.squeeze(0), faces, mode='torch').unsqueeze(0)
            tri_nrm = calc_norm_torch(verts_pos, faces)

            inputs_v = torch.cat([tri_cnt, tri_nrm], dim=-1) # [1, F, 3+3]
        return inputs_v
    
    def calc_new_mesh(self,
                    vertices,
                    faces,
                    z,
                    operators,
                    dfn_info,
                    img=None):
        """
        z: latent code
        """
        lu_solver, idxs, vals, rhs = operators
        # calc center of model
        cents = calc_cent(vertices, faces, mode='torch').float() # [1, V, 3]

        # normals
        norms = calc_norm_torch(vertices[None], faces).float() # [1, V, 3]

        # set inputs
        inputs = torch.cat([cents, norms], dim=-1)
        norms_v = calc_norm_torch(vertices[None], faces, at='vertex')  # [1, V, 3]
        
        # set source vertex
        input_target_v = torch.cat([vertices[None], norms_v], dim=-1).float()

        # get image feture
        img_feat = self.model.img_feat(img)

        # set inputs
        inputs_tri = torch.cat([inputs, img_feat.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)

        input_target_all = torch.cat([input_target_v, img_feat.unsqueeze(1).expand(-1, input_target_v.shape[1], -1)], dim=-1)

        with torch.no_grad():
            self.model.update_precomputes(dfn_info)
            
            pred_jacob = torch.zeros(z.shape[0], faces.shape[0], 3, 3).to(self.device)
            pred_vertices = torch.zeros(z.shape[0], vertices.shape[0], 3).to(self.device)
            
            for idx, z_i in enumerate (z):
                # decode to get mesh
                g_pred, z_iden = self.model.decode([inputs_tri.float(), input_target_all.float()], z_i[None])

                # solve for the mesh
                g_pred  = self.normalizer.inv_normalize(g_pred)
                g_pred  = nfr_utils.reconstruct_jacobians(g_pred, repr='matrix')
                pred_jacob[idx] = g_pred
                
                out_pred = self.myfunc(g_pred, lu_solver, idxs, vals, rhs.shape)
                pred_vertices[idx] = out_pred - out_pred.mean(axis=[0, 1], keepdim=True)
            
        return pred_vertices, g_pred, z_iden
    
    @torch.no_grad()
    def inference(self, 
                  vertices,
                  src_mesh,
                  tgt_mesh
                ):
        """
        ## Note: B (batch_size) is always 1
        Args:
            vertices (torch.tensor): animation of src mesh (time & vertex positions) [T, V, 3]
            src_mesh (trimesh.Trimesh): src face mesh with neutral face
            tgt_mesh (trimesh.Trimesh): tgt face mesh  with neutral face
        Return:
            pred_outputs (torch.tensor): [B, V, 3]
        """
        src_img = self.renderer.render_img(src_mesh).float().to(self.device)
        src_img_feat = self.get_img_feat(src_img)[None]
        
        src_dfn_info = nfr_utils.get_dfn_info(src_mesh, map_location=self.device) # neurtral face
        tgt_dfn_info = nfr_utils.get_dfn_info(tgt_mesh, map_location=self.device)

        src_vertices = vertices.to(self.device).float() # vertex with expression
        src_faces = torch.from_numpy(src_mesh.faces).to(self.device)

        tgt_verts = torch.from_numpy(tgt_mesh.vertices).to(self.device).float()
        tgt_faces = torch.from_numpy(tgt_mesh.faces).to(self.device)
        tgt_img = self.renderer.render_img(tgt_mesh).float().to(self.device)
        tgt_operators = self.get_mesh_operators(tgt_mesh)
        
        pred_outputs=[]
        pbar = tqdm(src_vertices)
        for src_v in pbar:
            inputs_v = self.get_inputs(src_v[None], src_faces)# [1, V, 3+3]

            ## get expression
            self.model.update_precomputes(src_dfn_info)
            pred_exp = self.model.encode(inputs_v, src_img.to(self.device), N_F=src_mesh.faces.shape[0])

            #pred_outputs, pred_jacobians, pred_id = self.calc_new_mesh(
            tmp, _, _ = self.calc_new_mesh(
                tgt_verts,
                tgt_faces,
                pred_exp,
                tgt_operators, 
                tgt_dfn_info, 
                tgt_img
            )
            pred_outputs.append(tmp)
        return torch.cat(pred_outputs)
    
    @torch.no_grad()
    def evaluate(self, batch, batch_process=True, return_all=False, stage=1, epoch=0, newid=None, mode=''):
        if stage == 1:
            ### train with ICT only + train mesh autoencoder only (learn rig space)
            if mode == 'invrig':
                return self.stage1_invrig(batch, newid, batch_process, return_all, epoch)
            else:
                return self.stage1_evaluate(batch, batch_process, return_all, epoch)
        
    def stage1_evaluate(self, batch, batch_process=True, return_all=False, stage=1, epoch=0):
        """
        ## Note: B (batch_size) is always 1
        Args:
            batch:
                audio_feat:    [B, W, 768] audio feature from wav2vec 2.0 -> not used here
                id_coeff:      [1, 128]    ICT-face model id coeff
                gt_rig_params: [B, W, 128] ICT-face model exp_coeff 
                template:      [B, V, 3] mesh vertices
                dfn_info (list): DiffusionNet information
                operators (list): mesh operators
                vertices:      [B, W, V, 3] sequence of mesh vertices (animation)
                faces:         [B, F, 3] mesh faces indices
                img:           [B, 256, 256, 3] rendered mesh image
            teacher_forcing (bool): used for Transformer model  -> not used here
            return_all (bool): if True, return loss and all predicted outputs
        Return:
            pred_outputs: [B, W, V, 3]
        """
        
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = batch.id_coeff.to(self.device).float()
        gt_rig_params = batch.gt_rig_params.to(self.device).float().squeeze(0)
        template = batch.template.to(self.device).float()
        gt_vertices = batch.vertices.to(self.device).float().squeeze(0) # [W, V, 3]
        faces = batch.faces.to(self.device).squeeze(0)
        img = batch.img.to(self.device).float()
        dfn_info=batch.get_dfn_info
        
        W, V, _ = gt_vertices.shape
        
        # expression encoder z
        pred_exp = torch.zeros((W, 128)).to(self.device)
        for t, verts in enumerate (gt_vertices):
            # vert&norm
            normals = calc_norm_torch(verts[None], faces, at='vert').to(self.device)
            inputs_v = torch.cat([verts[None], normals], dim=-1)

            with torch.no_grad():
                self.model.update_precomputes(dfn_info)
                pred_exp[t] = self.model.encode(inputs_v, img.to(self.device), N_F=faces.shape[0])
        
        operators = pickle.load(open(batch.operators, mode='rb'))
        
        # identity encoder + decoder
        pred_outputs, pred_jacobians, pred_id = self.calc_new_mesh(
                template[0],
                faces,
                pred_exp,
                operators, 
                dfn_info, 
                img
            )
        
        loss = {}
        
        gt_vertices_zm = gt_vertices - gt_vertices.mean(dim=1, keepdim=True) # because no translation included
        loss["recon_vDec"] = self.criterion(gt_vertices_zm, pred_outputs)
        
        gt_normals = calc_norm_torch(gt_vertices, faces, at='verts') #- [1, V, 3]
        pred_normals = calc_norm_torch(pred_outputs, faces, at='verts')
        loss["norm_vDec"] = self.criterion(gt_normals, pred_normals)
        
        gt_jacobians = get_jacobian_matrix(gt_vertices, faces, template, return_torch=True)
        loss["jacob_vDec"] = self.criterion(gt_jacobians, pred_jacobians)
        
        return loss, pred_outputs, None, pred_exp, pred_id, None
    
    def stage1_invrig(self, batch, newid, batch_process=True, return_all=False, stage=1, epoch=0):
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
            batch_process (bool): used for Transformer model  -> not used here
            return_all (bool): if True, return loss and all predicted outputs
        Return:
            pred_outputs: [B, W, V, 3]
        """
        (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data = batch
        
        #B, W, _ = audio_feat.shape
        #V = template.shape[1]
        
        ## Send to device --------------------------------------------------------------
        gt_id_coeff = id_coeff.to(self.device).float()
        gt_rig_params = gt_rig_params.to(self.device).float().squeeze(0)
        template = template.to(self.device).float()
        gt_vertices = vertices.to(self.device).float().squeeze(0) # [W, V, 3]
        faces = faces.to(self.device).squeeze(0)
        img = img.to(self.device).float()
        
        
        # get target mesh (neutral face and animated)
        (tgt_coeff, tgt_idx) = newid
        p_mode = 'face_only' if self.ict_face_only else 'fullhead'
        tgt_img = np.load(os.path.join(f'./ICT/precompute-synth-{p_mode}', f"{tgt_idx:03d}_img.npy")) # ----- [1, 256, 256, 3]
        tgt_img = torch.from_numpy(tgt_img).to(self.device).float()
        tgt_coeff = torch.from_numpy(tgt_coeff).to(self.device).float()
        tgt_gt_rig = gt_rig_params[:,:53].to(self.device).float()
        
        tgt_id_disps = torch.einsum('k,kls->ls', tgt_coeff, self.ict_face_model.id_basis)[:self.ict_face_model.v_idx]
        tgt_exp_disp = torch.einsum('jk,kls->jls', tgt_gt_rig, self.ict_face_model.exp_basis)[:,:self.ict_face_model.v_idx]
        
        tgt_template  = self.ict_neutral.to(self.device) + tgt_id_disps
        tgt_gt_vertices = tgt_template + tgt_exp_disp
        tgt_template = tgt_template.to(self.device).float()
        tgt_gt_vertices = tgt_gt_vertices.to(self.device).float()
        
        W, V, _ = gt_vertices.shape
        
        # expression encoder z from source
        pred_exp = torch.zeros((W, 128)).to(self.device)
        for t, verts in enumerate (gt_vertices):
            # vert&norm
            normals = calc_norm_torch(verts[None], faces, at='vert').to(self.device)
            inputs_v = torch.cat([verts[None], normals], dim=-1)

            with torch.no_grad():
                self.model.update_precomputes(dfn_info)
                pred_exp[t] = self.model.encode(inputs_v, img.to(self.device), N_F=faces.shape[0])
        
        operators = pickle.load(open(operators[0], mode='rb'))
        
        # identity encoder + decoder
        pred_outputs, pred_jacobians, pred_id = self.calc_new_mesh(
                #template[0],
                tgt_template,
                faces,
                pred_exp,
                operators, 
                dfn_info, 
                img
            )
        
        loss = {}
        pred_outputs = pred_outputs - pred_outputs.mean(dim=1, keepdim=True) # because no translation included
        loss["recon_vDec"] = self.criterion(tgt_gt_vertices, pred_outputs)
        
        tgt_gt_normals = calc_norm_torch(tgt_gt_vertices, faces, at='verts') #- [1, V, 3]
        pred_normals = calc_norm_torch(pred_outputs, faces, at='verts')
        loss["norm_vDec"] = self.criterion(tgt_gt_normals, pred_normals)
        
        tgt_gt_jacobians = get_jacobian_matrix(tgt_gt_vertices, faces, template, return_torch=True)
        loss["jacob_vDec"] = self.criterion(tgt_gt_jacobians, pred_jacobians)
        
        return loss, pred_outputs, None, pred_exp, pred_id, None

class Trainer():
    def __init__(self, opts):
        # set opts
        self.opts = opts

        self.set_seed(self.opts)
        self.device = self.opts.device
        
        if self.opts.NFR:
            # just in case for loading model from public NFR checkpoint
            self.model = NFR_helper(self.opts, self.device)
        else:
            self.model = NFS(self.opts, None, print_param=True).to(self.device)

            # load weight
            self.load_weight()
            
        self.criterion = nn.MSELoss()
        self.model.criterion = self.criterion
    
    def load_weight(self):
        if self.opts.ckpt:
            ckpt = glob.glob(os.path.join(self.opts.ckpt, "*_best.pth"))[0]
            print(f"Loading... {ckpt}")
            ckpt_dict = torch.load(ckpt)
            
            ## remove remaining precomputes in DiffusionNet
            del_key_list = ['mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X', 'grad_Y', 'faces']
            if "ckpt_stage1" in self.opts.ckpt or self.opts.stage1:
                del_key_list.append('audio_encoder')
            ckpt_dict = del_key(ckpt_dict, del_key_list)
            
            # self.model.load_state_dict(ckpt_dict,strict=False)
            self.model.load_state_dict(ckpt_dict)
    
    def get_mesh_standardization(self, mesh_data="voca", base_dir="./mesh_utils"):
        """Apply mesh standardization with pre-calculated vertex indices and face indices
        Args:
            mesh_data (str)
            base_dir (str)
        Return:
            info_dict (dict)
        """
        # load info
        basename = os.path.join(base_dir, mesh_data, "standardization.npy")
        info_dict = np.load(basename, allow_pickle=True).item()
        return info_dict
    
    def test_stage1(self):
        # define loss lamdba -------------------------------------------------------------------------------------
        self.loss_lambda = {
            "recon": 1.,
            "norm":  1.,
            "jacob": 1.,
            "temp":  1.,
        }
        
        # define dataset -----------------------------------------------------------------------------------------
        # self.opts.selection = 2
        self.test_dataset = NFSDataset(self.opts, is_train=False, is_valid=False, return_audio_dir=True)
       
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
#         testsampler = MeshSampler_coarse(self.test_dataset.len_list, batch_size=1, shuffle=False, n_sampling=True, n_=30)
#         self.test_dataloader = torch.utils.data.DataLoader(
#             self.test_dataset, 
#             batch_sampler=testsampler, 
#             collate_fn=partial(collate_wrapper, device=self.opts.device), 
#             num_workers=0)
        #len_test = len(self.test_dataset)
        len_test = len(self.test_dataloader)
        
        assert len_test > 0, f"test Dataset: {len_test}"
        print(f"test Dataset: {len_test}")
        
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        s_num = self.opts.selection
        print(f"selection: {s_num}")
        if self.opts.on_ict:
            if s_num > 2:
                raise ValueError('on_ict only available on selection 2')
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, f"eval-retarget-select_{s_num:02d}")
        os.makedirs(self.opts.log_dir, exist_ok=True)
                            
        # self logger
        ckpt_name = self.opts.ckpt.split('/')[-1] if not self.opts.NFR else 'pretrained_NFR'
        if self.opts.on_ict:
            logger_file  = os.path.join(self.opts.log_dir, f"{ckpt_name}-on_ict-log.txt")
        else:
            logger_file  = os.path.join(self.opts.log_dir, f"{ckpt_name}-log.txt")
        
        if os.path.exists(logger_file):
            self.logger = open(logger_file, 'a')
            
            import datetime
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d-%H-%M-%S")
            self.logger.write(f'\n[new log added]: {now}\n')
        else:
            self.logger = open(logger_file, 'w')
        self.logger.write(f'data selection: {self.opts.selection:02d}\n')
        
        print(f'Saving log at: {self.opts.log_dir}')
        print(self.test_dataset.get_data_config())
        # print(testsampler.get_sampler_config())
        #self.logger.write(self.test_dataset.get_data_config())
        
        if self.opts.save_vert:
            save_vert_path = os.path.join(self.opts.log_dir, ckpt_name, self.opts.feat_level)
            os.makedirs(save_vert_path, exist_ok=True)
            save_rigs_path = os.path.join(self.opts.log_dir, ckpt_name, self.opts.feat_level+'_rig')
            os.makedirs(save_rigs_path, exist_ok=True)
            
            save_rig_img_path = os.path.join(self.opts.log_dir, ckpt_name, 'rig_img')
            os.makedirs(save_rig_img_path, exist_ok=True)
            
        metric={}
        
        self.model.eval()
        print(f"Evaluation ... [stage1]")
        ict_face_model = self.test_dataset.ict_face_model
        
        global_step = 0
        recon_vDec = []
        for index, batch in tqdm(enumerate(self.test_dataloader), total=len_test):
            # ------------------------------------------------------------------------------------------------
            #(_, _, _, _, _, _, vertices, faces, _), mesh_data = batch
            
            audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, v_normal, faces, img, mesh_data, audio_path = batch
            # data, mesh_data, audio_path = batch
            
            #gt_id_coeff = batch.id_coeff.cpu()
#             gt_rig_params = batch.gt_rig_params.cpu()
#             vertices = batch.vertices.cpu().squeeze(0)
#             template = batch.template
#             faces = batch.faces.cpu()
#             audio_path = batch.audio_path
            data = (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img)
            from easydict import EasyDict as edict
            batch = edict()
            batch.audio_feat=audio_feat
            batch.id_coeff=id_coeff
            batch.gt_rig_params=gt_rig_params
            batch.template=template
            batch.get_dfn_info=dfn_info
            batch.operators=operators[0]
            batch.vertices=vertices
            batch.faces=faces[0]
            batch.img=img
            batch.mesh_data=mesh_data
            mesh_data = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
            with torch.no_grad():
                loss_dict, pred_vertices, _, pred_exp_coeff, pred_id_coeff, pred_seg = self.model.evaluate(
                    batch, \
                    batch_process=False, \
                    return_all=True, \
                    stage=1, \
                    epoch=500
                )
                
            if self.opts.on_ict:
                exp_b = torch.from_numpy(ict_face_model.exp_basis)
                exp_disp = torch.einsum('jk,kls->jls', pred_exp_coeff[:, :53], exp_b.to(self.device).float())[:,:ict_face_model.v_idx]
                pred_outputs = template.to(self.device).float() + exp_disp
                gt_vertices = vertices.to(self.device).float().squeeze(0)
                loss_dict["recon_vDec"] = self.criterion(gt_vertices, pred_outputs)
                
            # get total loss
            for key, value in loss_dict.items():
                if key in metric.keys():
                    metric[key] += value
                else:
                    metric[key] = value
            recon_vDec.append(loss_dict["recon_vDec"].detach().cpu().numpy())
            
            if self.opts.save_vert:
                try:
                    _, feat_lvl, id_sent = audio_path[0].split('/wav2vec2')[-1].split('/')
                    if self.opts.selection <= 2:
                        id_ = audio_path[0].split('/wav2vec2')[0].split('/')[-1]
                        id_sent = id_ +'_'+ id_sent
                except:
                    if mesh_data == 'mf':
                        if self.opts.selection:
                            id_sent='-'.join(audio_path[0].split('/')[-2:])
                        else:
                            id_sent = '-'.join(audio_path[0].split('/')[-3:-1])
                    elif mesh_data =='ict':
                        id_sent = audio_path[0]
                    else:
                        import pdb;pdb.set_trace()
                        
                plt.imshow(np.r_[
                    gt_rig_params.squeeze(0).cpu().numpy(), 
                    np.zeros([1,128]),
                    pred_exp_coeff.cpu().numpy()
                ])
                plt.savefig(f'{save_rig_img_path}/eval-rig_{index:04d}.png')
                plt.close()
                save_vert_file = os.path.join(save_vert_path, id_sent)
                np.save(save_vert_file, pred_vertices.detach().cpu().numpy())
                save_rigs_file = os.path.join(save_rigs_path, id_sent)
                np.save(save_rigs_file, pred_exp_coeff.detach().cpu().numpy())
            # ------------------------------------------------------------------------------------------------
            global_step += 1
            if self.opts.debug:
                break
            # ------------------------------------------------------------------------------------------------
        recon_vDec = np.array(recon_vDec)        
        recon_vDec_mu = np.mean(recon_vDec)
        recon_vDec_std = np.std(recon_vDec)
        
        tmp_log = f"recon_vDec_mu: {recon_vDec_mu}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        tmp_log = f"recon_vDec_std: {recon_vDec_std}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        
        
        for key, value in metric.items():
            tmp_log = f"{key}: {value/len_test}"
            print(tmp_log)
            self.logger.write(tmp_log+'\n')
            
    
    def test_stage1_inv_rig(self):
        # define loss lamdba -------------------------------------------------------------------------------------
        self.loss_lambda = {
            "recon": 1.,
            "norm":  1.,
            "jacob": 1.,
            "temp":  1.,
        }
        
        # define dataset -----------------------------------------------------------------------------------------
        self.opts.selection = 2
        self.test_dataset = InvRigDataset(self.opts, is_train=False, is_valid=False, print_config=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        len_test = len(self.test_dataset)
        
        assert len_test > 0, f"test Dataset: {len_test}"
        print(f"test Dataset: {len_test}")
        
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        s_num = self.opts.selection
        print(f"selection: {s_num}")
        self.opts.log_dir = os.path.join(self.opts.log_dir, f"eval-inv_rig-select_{s_num:02d}")
        os.makedirs(self.opts.log_dir, exist_ok=True)
                            
        # self logger
        ckpt_name = self.opts.ckpt.split('/')[-1] if not self.opts.NFR else 'pretrained_NFR'
        self.logger = open(os.path.join(self.opts.log_dir, f"{ckpt_name}-log.txt"), 'w')
        self.logger.write(f'data selection: {self.opts.selection:02d}\n')
        
        metric={
            "max_LVE": 0,
            "max_LVE_std": 0,
            "mean_LVE": 0,
        }
        
        if not self.opts.NFR:
            self.model.eval()
        print(f"Evaluation ... [stage1]")
        ict_model = self.test_dataset.ict_face_model
        
        global_step = 0
        recon_vDec = []
        for id_idx, iden_vec in enumerate (self.test_dataset.iden_vecs):
            for index, batch in tqdm(enumerate(self.test_dataloader), total=len_test):
                # ------------------------------------------------------------------------------------------------
                (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data = batch
                
                if (id_coeff[:,:100]==torch.from_numpy(iden_vec).float()).all():
                    continue
                    
                vertices = vertices.squeeze(0) # [1, T, V, 3] -> [T, V, 3]

                with torch.no_grad():
                    loss_dict, pred_vertices, _, pred_exp_coeff, pred_id_coeff, pred_seg = self.model.evaluate(
                        batch, \
                        batch_process=False, \
                        return_all=True, \
                        stage=1, \
                        epoch=500, \
                        newid=(iden_vec,id_idx), \
                        mode='invrig',
                    )

                if self.opts.on_ict:
                    exp_disp = torch.einsum('jk,kls->jls', pred_exp_coeff[:, :53], ict_face_model.exp_basis.to(self.device).float())[:,:ict_face_model.v_idx]
                    pred_outputs = template.to(self.device).float() + exp_disp
                    gt_vertices = vertices.to(self.device).float().squeeze(0)
                    loss_dict["recon_vDec"] = nn.L1Loss()(gt_vertices, pred_outputs)
                    
                # get total loss
                for key, value in loss_dict.items():
                    if key in metric.keys():
                        metric[key] += value
                    else:
                        metric[key] = value
                recon_vDec.append(loss_dict["recon_vDec"].detach().cpu().numpy())

                # ------------------------------------------------------------------------------------------------
                global_step += 1
                if self.opts.debug:
                    break
                # ------------------------------------------------------------------------------------------------
        recon_vDec = np.array(recon_vDec)        
        recon_vDec_mu = np.mean(recon_vDec)
        recon_vDec_std = np.std(recon_vDec)
        
        tmp_log = f"recon_vDec_mu: {recon_vDec_mu}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        tmp_log = f"recon_vDec_std: {recon_vDec_std}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        
        for key, value in metric.items():
            tmp_log = f"{key}: {value/(len_test*self.test_dataset.iden_vecs.shape[0])}"
            print(tmp_log)
            self.logger.write(tmp_log+'\n')
    
    
    def test_segment(self):
        # define dataset -----------------------------------------------------------------------------------------
        self.opts.selection = 2
        self.test_dataset = NFSDataset(self.opts, is_train=False, is_valid=False)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)
        len_test = len(self.test_dataset)
        
        assert len_test > 0, f"test Dataset: {len_test}"
        print(f"test Dataset: {len_test}")
        
        # make logdir --------------------------------------------------------------------------------------------
        os.makedirs(self.opts.log_dir, exist_ok=True)
        
        s_num = self.opts.selection
        print(f"selection: {s_num}")
        if self.opts.on_ict and s_num != 2:
            raise ValueError('on_ict only available on selection 2')
        
        self.opts.log_dir = os.path.join(self.opts.log_dir, f"eval-segment-select_{s_num:02d}")
        os.makedirs(self.opts.log_dir, exist_ok=True)
                            
        # self logger
        ckpt_name = self.opts.ckpt.split('/')[-1] if not self.opts.NFR else 'pretrained_NFR'
        if self.opts.on_ict:
            logger_file  = os.path.join(self.opts.log_dir, f"{ckpt_name}-on_ict-log.txt")
        else:
            logger_file  = os.path.join(self.opts.log_dir, f"{ckpt_name}-log.txt")
        
        if os.path.exists(logger_file):
            self.logger = open(logger_file, 'a')
            
            import datetime
            now = datetime.datetime.now()
            now = now.strftime("%Y-%m-%d-%H-%M-%S")
            self.logger.write(f'\n[new log added]: {now}\n')
        else:
            self.logger = open(logger_file, 'w')
        self.logger.write(f'data selection: {self.opts.selection:02d}\n')
        
        print(f'Saving log at: {self.opts.log_dir}')
        print(self.test_dataset.get_data_config())
        #self.logger.write(self.test_dataset.get_data_config())
        
        metric={}
        
        self.model.eval()
        print(f"Evaluation ... [stage1]")
        ict_face_model = self.test_dataset.ict_face_model
        
        global_step = 0
        recon_vDec = []
        precision = []
        ict_vert_segment = np.load('./utils/ict/ICT_segment_onehot.npy') # [V, 20]
        
        for index, batch in tqdm(enumerate(self.test_dataloader), total=len_test):
            # ------------------------------------------------------------------------------------------------
            #(_, _, _, _, _, _, vertices, faces, _), mesh_data = batch
            (audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data = batch
            vertices = vertices.squeeze(0) # [1, T, V, 3] -> [T, V, 3]
            
            with torch.no_grad():
                loss_dict, pred_vertices, _, pred_exp_coeff, pred_id_coeff, pred_seg = self.model.evaluate(
                    batch, \
                    batch_process=False, \
                    return_all=True, \
                    stage=1, \
                    epoch=500
                )
            
            
            pred_seg_label = pred_seg.argmax(-1).detach().cpu().numpy().squeeze(0) # [1, V]
            np_eye = np.eye(20)
            pred_seg_one_hot = np_eye[pred_seg_label] # [V, 20]
            pred_seg_TP = ict_vert_segment * pred_seg_one_hot # [V, 20]
            
            curr_precision = pred_seg_TP.sum(0) / pred_seg_one_hot.sum(0) # # [1, 20]
            curr_precision = curr_precision.mean() # mean over all segments
            precision.append(curr_precision)
            # for idx in range(20):
                
            if self.opts.on_ict:
                exp_disp = torch.einsum('jk,kls->jls', pred_exp_coeff[:, :53], ict_face_model.exp_basis.to(self.device).float())[:,:ict_face_model.v_idx]
                pred_outputs = template.to(self.device).float() + exp_disp
                gt_vertices = vertices.to(self.device).float().squeeze(0)
                loss_dict["recon_vDec"] = nn.L1Loss()(gt_vertices, pred_outputs)
                
            # get total loss
            for key, value in loss_dict.items():
                if key in metric.keys():
                    metric[key] += value
                else:
                    metric[key] = value
            recon_vDec.append(loss_dict["recon_vDec"].detach().cpu().numpy())
            
            # ------------------------------------------------------------------------------------------------
            global_step += 1
            if self.opts.debug:
                break
            # ------------------------------------------------------------------------------------------------
        recon_vDec = np.array(recon_vDec)
        recon_vDec_mu = np.mean(recon_vDec)
        recon_vDec_std = np.std(recon_vDec)

        tmp_log = f"recon_vDec_mu: {recon_vDec_mu}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        tmp_log = f"recon_vDec_std: {recon_vDec_std}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        
        precision = np.array(precision)
        precision_mu = np.mean(precision)
        precision_std = np.std(precision)
        
        tmp_log = f"precision_mu: {precision_mu}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        tmp_log = f"precision_std: {precision_std}"
        print(tmp_log)
        self.logger.write(tmp_log+'\n')
        
        for key, value in metric.items():
            tmp_log = f"{key}: {value/len_test}"
            print(tmp_log)
            self.logger.write(tmp_log+'\n')

    @staticmethod
    def log_loss(writer, loss_dict, step, counter=None):
        if counter:
            N = counter
        else:
            N = 1
        for key, value in loss_dict.items():
            writer.add_scalar(f'{key}', value/N, step)

    @staticmethod
    def set_seed(opts):
        # set seed
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def dump_yaml(yaml_file_path, opts):
        with open(yaml_file_path, 'w') as f: 
            yaml.dump(vars(opts), f, sort_keys=False)

    @torch.no_grad()
    def render_output(self, inputs, src_mesh, tgt_mesh, wav_path, mode='mesh', y_rot=0, batch_process=True):
        if mode not in ['audio','mesh']:
            raise NotImplementedError(f"given mode: {mode}")
            
        self.model.eval()
        with torch.no_grad():
            pred_vertices = self.model.predict(inputs, src_mesh, tgt_mesh, mode=mode, batch_process=batch_process)
            pred_vertices = pred_vertices.detach().cpu().numpy()
        pred_vertices = pred_vertices * 0.6
        
        ckpt_name = self.opts.ckpt.split('/')[-1]
        render_w_audio(pred_vertices, tgt_mesh.faces, savedir='_tmp',y_rot=y_rot, savename=f'SIGA-{ckpt_name}-{id_}_{sent}', audio_fn=wav_path)
        render_wo_audio(pred_vertices, tgt_mesh.faces, savedir='_tmp',y_rot=y_rot, savename=f'SIGA-{ckpt_name}-{id_}_{sent}-no_audio')

"""
tensorboard --logdir ./ckpt_stage1 --port 6789
"""
            
if __name__ == "__main__":
    # argparse configs
    opts = Options()
    
    # get train configs from ckpt (yaml)
    if not opts.NFR:
        opts.config = os.path.join(opts.ckpt, "train_opts.yml")
    opts_yaml = yaml.load(open(opts.config), Loader=yaml.FullLoader)
    # update with argparse configs
    opts_ = vars(opts)
    opts_yaml.update(opts_)
    opts = argparse.Namespace(**opts_yaml)
    
    opts.data_rand_trans=False
    opts.data_rand_scale=False
    
    if opts.stage1:
        opts.log_dir = 'evaluate/stage1'
    if opts.stage11:
        opts.log_dir = 'evaluate/stage11'
        
    trainer = Trainer(opts)
    
    if opts.stage1:
        if opts.inv_rig:
            trainer.test_stage1_inv_rig()
        else:
            trainer.test_stage1()
    
    if opts.segment:
        trainer.test_segment()
    
