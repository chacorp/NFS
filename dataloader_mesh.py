import os
import glob
import numpy as np
import torch
import torch.utils.data as data
import random
import pickle
import trimesh
from functools import partial

from utils import (
    ICT_face_model, 
    procrustes_LDM, 
    plot_image_array, 
    calc_norm_torch
)
from utils.keys import get_data_splits, get_identity_num, ICT_KEYS, DATA_KEYS, KEYS
from utils.remesh_utils import map_vertices, decimate_mesh_vertex
from utils.mesh_utils import get_dfn_info2, get_mesh_operators


def random_ict_exp_coeff(N=1000, min_ones=1, max_ones=5):
    identity = np.eye(53)

    num_ones_per_row = np.random.randint(min_ones, max_ones + 1, size=N)

    exp_coeffs = np.array([np.sum(identity[np.random.choice(53, num_ones, replace=False)], axis=0)
                    for num_ones in num_ones_per_row], dtype=int)
    return exp_coeffs

class NFSDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 ict_basedir='/data/ICT-audio2face/split_set/', 
                 mf_basedir="/data/multiface/audio2face",
                 return_audio_dir=False,
                 audio_feat_type: str ='wav2vec2',
                 audio_feat_level: str = "05",
                 is_train=False,
                 is_valid=False,
                 window_size=8, # batch size
                 ict_face_only=True,
                 device='cpu',
                 print_config=False,
                ):
        super().__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        
        self.device = device
        self.audio_feat_level = audio_feat_level
        self.audio_feat_type = audio_feat_type
        self.return_audio_dir = return_audio_dir
        
        if self.opts is None:
            self.use_decimate = False
            self.WS = window_size
            self.ict_face_only = ict_face_only
        else:
            self.use_decimate = self.opts.use_decimate
            self.WS = self.opts.window_size
            self.ict_face_only = self.opts.ict_face_only
         
        self.select_data(self.opts.selection if opts is not None else 0)
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
            
        ## precomputes --------------------------------------------------------------------
        p_mode = 'face_only' if self.ict_face_only else 'fullhead'
        self.ict_data_split, self.voca_data_split, self.biwi_data_split, self.mf_data_split = get_data_splits()
        self.identity_num = get_identity_num()
        
        self.only_ict = False
        ## ICT-facekit --------------------------------------------------------------------
        self.ict_templates_path = './ICT/templates'
        self.iden_vecs = np.load('./data/ICT_live_100/iden_vecs.npy')
        self.expression_vecs = np.load(f'./data/ICT_live_100/expression_vecs_{self.mode}.npy')
        
        if self.mode != 'test':
            newvecs = np.load('./ict_face_pt/random_expression_vecs.npy')
            id_zero = np.zeros([1, 100])
            id_vecs = np.eye(100, 100)*3.0
            self.expression_vecs = np.r_[self.expression_vecs, newvecs]
            self.iden_vecs = np.r_[self.iden_vecs, id_zero, id_vecs]
        
        self.ict_face_model = ICT_face_model(face_only=False)
        self.ict_precompute_path = f'./ICT/precompute-fullhead'
        self.ict_precompute_path_synth = f'./ICT/precompute-synth-fullhead'
        
        self.ict_face_model_fo = ICT_face_model(face_only=True)
        self.ict_precompute_path_fo = './ICT/precompute-face_only'
        self.ict_precompute_path_fo_synth = './ICT/precompute-synth-face_only'
        
        self.ict_basedir = os.path.join(ict_basedir, self.mode)
        self.ict_audio_wav = sorted(glob.glob(f'{self.ict_basedir}/*/wav/*.wav'))
        
        if self.opts.seg_dim == 20:
            self.ict_vert_segment = torch.from_numpy(np.load('./utils/ict/ICT_segment_onehot.npy'))
        elif self.opts.seg_dim == 24:
            self.ict_vert_segment = torch.from_numpy(np.load('./utils/ict/ICT_segment_onehot_24.npy'))
        elif self.opts.seg_dim == 14:
            self.ict_vert_segment = torch.from_numpy(np.load('./utils/ict/ICT_segment_onehot_14.npy'))
        elif self.opts.seg_dim == 6:
            self.ict_vert_segment = torch.from_numpy(np.load('./utils/ict/ICT_segment_onehot_06.npy'))
        else:
            raise NotImplementedError(f"no segment map for seg_dim: {self.opts.seg_dim}")
        ## --------------------------------------------------------------------------------
        
            
        ## Multiface ----------------------------------------------------------------------
        if self.use_mf_SEN or self.use_mf_ROM:
            self.mf_trimesh = trimesh.load('./utils/mf/mf_aligned_mean.obj', process=False, maintain_order=True)
            self.mf_precompute_path = os.path.join(mf_basedir, 'precomputes_std')
            self.mf_std = np.load(f'./utils/mf/standardization.npy', allow_pickle=True).item()
            
            self.mf_basedir = os.path.join(mf_basedir)
            self.mf_id_v = self.get_mf_id_verts(self.mf_basedir)
            
            self.mf_audio_wav = []
            self.mf_ROM = []
            for id_ in self.mf_data_split[self.mode]:
                self.mf_audio_wav+=sorted(glob.glob(f'{self.mf_basedir}/wav/{id_}*.wav'))
                self.mf_ROM +=sorted(glob.glob(f'{self.mf_basedir}/vertices_npy_exp/{id_}/*.npy'))
        ## --------------------------------------------------------------------------------
        
        self.len_ict_synth = 0
        self.len_ict_real = 0
        self.len_mfROM = 0
        self.len_mfSEN = 0
        self.total_len = 0
        
        self.len_list = []
        if self.use_ict_real:
            self.len_ict_real = len(self.ict_audio_wav)
            self.len_list.append([self.len_ict_real, self.get_ICTcapture, torch.tensor(0)])
        if self.use_ict_synth:
            self.len_ict_synth = self.expression_vecs.shape[0]
            self.len_list.append([self.len_ict_synth, self.get_ICTsynthetic, torch.tensor(0)])
        if self.use_mf_SEN:
            self.len_mfSEN = len(self.mf_audio_wav)
            self.len_list.append([self.len_mfSEN, self.get_multiface_SEN, torch.tensor(3)])
        if self.use_mf_ROM:
            self.len_mfROM = len(self.mf_ROM)
            self.len_list.append([self.len_mfROM, self.get_multiface_ROM, torch.tensor(3)])
        
        self.total_len = self.len_ict_real+self.len_ict_synth+self.len_mfSEN+self.len_mfROM

        if print_config:
            print(self.get_data_config())
        
    def get_data_config(self):
        text = "========[NewStage1Dataset]========\n"
        text += f"mode: {self.mode}\n"
        text += f"[ICT-synth data]: {self.len_ict_synth}\n"
        text += f"[ICT-capture data]: {self.len_ict_real}\n"
        text += f"[multiface data]: {self.len_mfSEN+self.len_mfROM}\n"
        text += f"-----------------------\n"
        text += f"[total data]: {self.total_len}\n"
        text += "===============================\n"
        return text
    
    def __len__(self):
        return self.total_len
            
    def select_data(self, selection):
        self.use_ict_real=False
        self.use_ict_synth=False
        self.use_mf_SEN=False
        self.use_mf_ROM=False
        
        if selection == 0: ## ICT-synthetic
            self.use_ict_synth=True
        
        elif selection == 1: ## ICT-capture
            self.use_ict_real=True
        
        elif selection == 2: ## ICT-all (ICT-capture + ICT-synthetic)
            self.use_ict_real=True
            self.use_ict_synth=True
        
        elif selection == 5: ## Multiface_SEN
            self.use_mf_SEN=True
        
        elif selection == 6: ## Multiface_ROM
            self.use_mf_ROM=True
        
        elif selection == 7: ## Multiface_SEN + Multiface_ROM
            self.use_mf_ROM=True
            self.use_mf_SEN=True
        
        elif selection == 20: ## ICT-synth (0) + Multiface_SEN (4) + Multiface_ROM (5)
            self.use_ict_synth=True
            self.use_mf_ROM=True
            self.use_mf_SEN=True

        elif selection == 21: ## all
            self.use_ict_real=True
            self.use_ict_synth=True
            self.use_mf_ROM=True
            self.use_mf_SEN=True
        else:
            raise NotImplementedError(f"no selection: {selection}")
          
    def get_ICTcapture(self, index):
        f_splits  = self.ict_audio_wav[index].split('/')
        id_, sent = f_splits[-3], f_splits[-1].split('.')[0]
        
        # get id_coeff
        id_coeff  = torch.load(os.path.join(self.ict_templates_path, f"{id_}_coeffs.pt"))
        id_coeff  = torch.cat([id_coeff, torch.zeros(28)])
        
        # get exp_coeff + slice w/ window
        exp_path = os.path.join(self.ict_basedir, id_, 'rig_param', f"{sent}.npy")
        exp_coeff = torch.from_numpy(np.load(exp_path))
        
        T = exp_coeff.shape[0]
        if self.mode == 'test':
            self.WS = T
            slice_idx = 0
        else:
            slice_idx = random.randint(0, T-self.WS-1)
            
        exp_coeff = exp_coeff[slice_idx:slice_idx+self.WS]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(self.WS, 75)], dim=-1)
        
        # get template vertices and face indices (neutral face)
        vertices, template, _ = self.ict_face_model.apply_coeffs(
            id_coeff[:100].numpy(), 
            exp_coeff[:,:53].numpy(), 
            return_all=True
        )
        vertices = torch.from_numpy(vertices).float()
        template = torch.from_numpy(template[0]).float()
        faces = torch.from_numpy(self.ict_face_model.faces).long()
        
        # get template dfn_info (neutral face)
        dfn_info  = pickle.load(open(os.path.join(
            self.ict_precompute_path,
            f"{id_}_dfn_info.pkl"
        ), 'rb'))
        operators = os.path.join(
            self.ict_precompute_path,
            f"{id_}_operators.pkl"
        )
        
        img = np.load(os.path.join(self.ict_precompute_path, f"{id_}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        # get audio feature + slice w/ window
        dummy = torch.zeros(self.WS, 768)
        
        v_normal = calc_norm_torch(vertices, faces, at='v').float()
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, sent
        
    def get_ICTsynthetic(self, index):
        e_index   = index % self.expression_vecs.shape[0]
        if self.expression_vecs.shape[0] - e_index < self.WS:
            e_index = self.expression_vecs.shape[0] - self.WS
        id_idx     = np.random.randint(0, self.iden_vecs.shape[0])
        
        # get id_coeff
        id_coeff  = torch.from_numpy(self.iden_vecs[id_idx]).float() #------------------------------------ [100]
        id_coeff  = torch.cat([id_coeff, torch.zeros(28)]) # --------------------------------------------- [128]
        
        exp_coeff = torch.from_numpy(self.expression_vecs).float()[e_index:e_index+self.WS] # ------------ [W, 53]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(self.WS, 75)], dim=-1) # --------------------------- [W, 128]
        
        # get template vertices and face indices (neutral face)
        if not self.ict_face_only:
            if random.random() > 0.5:
                ict_model = self.ict_face_model
                ict_path_synth = self.ict_precompute_path_synth
            else:
                ict_model = self.ict_face_model_fo
                ict_path_synth = self.ict_precompute_path_fo_synth
        
        vertices, template, _ = ict_model.apply_coeffs(
            id_coeff[:100].numpy(), 
            exp_coeff[:,:53].numpy(), 
            return_all=True
        )
        vertices = torch.from_numpy(vertices).float()
        template = torch.from_numpy(template[0]).float()
        faces = torch.from_numpy(ict_model.faces).long()
        
        # get template dfn_info (neutral face)
        dfn_info  = pickle.load(open(os.path.join(ict_path_synth, f"{id_idx:03d}_dfn_info.pkl"), 'rb'))
        operators = os.path.join(ict_path_synth, f"{id_idx:03d}_operators.pkl")
        
        img = np.load(os.path.join(ict_path_synth, f"{id_idx:03d}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        dummy = torch.zeros(self.WS, 768)
        
        v_normal = calc_norm_torch(vertices, faces, at='v')

        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, ''
        
    def get_multiface_SEN(self, index):
        f_splits = self.mf_audio_wav[index].split('/')
        sent = f_splits[-1].split('.')[0]
        id_ = sent.split('-SEN')[0]
        
        # get audio feature + slice w/ window
        audio_path = os.path.join(
            self.mf_basedir, 
            self.audio_feat_type, 
            self.audio_feat_level, 
            f"{sent}.npy"
        )
        audio_feat_full = np.load(audio_path)
        
        T = audio_feat_full.shape[0]
        slice_idx = random.randint(0, T-self.WS)

        dummy = torch.zeros(self.WS, 768)

        # get template vertices (neutral face)
        id_index = self.mf_data_split[self.mode].index(id_)
        template = self.mf_id_v[id_index]
        faces = self.mf_std['new_f']
        
        # get template dfn_info (neutral face)
        dfn_info = pickle.load(open(os.path.join(
            self.mf_precompute_path,
            f"{id_}_dfn_info.pkl"
        ), 'rb'))
        operators= os.path.join(
            self.mf_precompute_path,
            f"{id_}_operators.pkl"
        )
        
        img = np.load(os.path.join(self.mf_precompute_path, f"{id_}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        # get animation vertices (25fps)
        npy_files_dir = os.path.join(self.mf_basedir, 'vertices_npy', f"{sent}")
        vertices, v0 = self.load_mf_SEN_verts(npy_files_dir, sent, slice_idx, self.WS)
        
        # align
        v0 = v0[self.mf_std['v_idx']]
        vertices = vertices[:, self.mf_std['v_idx']]
        R1, t1, s1 = procrustes_LDM(v0, template, mode='np')
        vertices = (s1*vertices)@R1.T+t1
        
        template = torch.from_numpy(template).float()
        vertices = torch.from_numpy(vertices).float()

        # get exp_coeff  (no GT == zeros!)
        exp_coeff= torch.zeros(self.WS, 128).float()
        
        # get id_coeff (no GT == zeros!)
        id_coeff = torch.zeros(128).float()
        
        v_normal = calc_norm_torch(vertices, faces, at='v').float()

        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, audio_path
    
    def get_multiface_ROM(self, index):
        """
        always use neutral face at 0th 
        """
        
        npy_file = self.mf_ROM[index]
        f_splits = npy_file.split('/')
        ROM = f_splits[-1].split('.')[0]
        id_ = f_splits[-2]
        
        id_index = self.mf_data_split[self.mode].index(id_)
        template = self.mf_id_v[id_index]
        faces = self.mf_std['new_f']
        
        npy_files_dir = npy_file.replace('.npy', '')
        vertices, v0 = self.load_mf_ROM_verts(npy_files_dir, WS=self.WS)

        ## align
        v0 = v0[self.mf_std['v_idx']]
        vertices = vertices[:, self.mf_std['v_idx']]
        R1, t1, s1 = procrustes_LDM(v0, template, mode='np')
        vertices = (s1*vertices)@R1.T+t1

        # get template dfn_info (neutral face)
        dfn_info  = pickle.load(open(os.path.join(
            self.mf_precompute_path, 
            f"{id_}_dfn_info.pkl"
        ), 'rb'))
        operators = os.path.join(
            self.mf_precompute_path,
            f"{id_}_operators.pkl"
        )
        img = np.load(os.path.join(self.mf_precompute_path, f"{id_}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        vertices = torch.from_numpy(vertices).float()
        template = torch.from_numpy(template).float()
        
        # get id_coeff and exp_coeff (no GT == zeros!)
        id_coeff = torch.zeros(128)
        exp_coeff = torch.zeros(self.WS, 128)
        dummy = torch.zeros(self.WS, 768)
        
        v_normal = calc_norm_torch(vertices, faces, at='v')
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, npy_file    
    
    def get_mf_id_verts(self, basedir):
        #id_verts_dir = f'{basedir}/id_verts.npy'
        id_verts_dir = os.path.join(basedir, f'id_verts_{self.mode}.npy')
        
        if os.path.exists(id_verts_dir):
            id_verts = np.load(id_verts_dir)
        else:
            id_verts=[]
            for id_ in self.mf_data_split[self.mode]:
                template_= trimesh.load(os.path.join(self.mf_precompute_path, f"{id_}_mesh.obj"), process=False, maintain_order=True)
                id_verts.append(template_.vertices)
            id_verts = np.array(id_verts)
            print(f'saving cache at: {id_verts_dir}')
            np.save(id_verts_dir, id_verts)
            
        return id_verts
    
    def load_mf_SEN_verts(self, basedir, SENT, slice_idx, WS):
        if os.path.exists(basedir):
            vertices = np.zeros((WS, 7306, 3))
            for i, j in enumerate(range(slice_idx,slice_idx+WS)):
                exp_verts_npy_dir = os.path.join(basedir, f"{j:04d}.npy")
                try:
                    vertices[i] = np.load(exp_verts_npy_dir)
                except:
                    raise ValueError(f'something is wrong! {exp_verts_npy_dir}')
            # for alignment
            v0 = np.load(os.path.join(basedir, f"{0:04d}.npy"))
        else:
            vertices = np.load(os.path.join(basedir, 'vertices_npy', f"{SENT}.npy"))
            vertices = vertices.reshape(vertices.shape[0], -1, 3)
            
            # for alignment
            v0 = vertices[0]
            print(f'saving each frame at: {basedir}')
            os.makedirs(basedir, exist_ok=True)
            for j, verts_npy in enumerate(vertices):
                exp_verts_npy_dir = os.path.join(basedir, f"{j:04d}")
                np.save(exp_verts_npy_dir, verts_npy)
            vertices = vertices[slice_idx:slice_idx+WS] #------------------------------------ [W, V, 3]
        
        return vertices, v0
    
    def load_mf_ROM_verts(self, basedir, WS):
        if os.path.exists(basedir):
            Frames = len(glob.glob(os.path.join(basedir, "*.npy")))
            if self.mode != 'test': 
                try:
                    slice_idx = self.get_slice_idx(Frames, WS)
                except:
                    raise ValueError(f'something is wrong! {Frames}')
            else:
                slice_idx = 0
                WS = Frames
            
            vertices = np.zeros((WS, 7306, 3))
            for i, j in enumerate(range(slice_idx,slice_idx+WS)):
                exp_verts_npy_dir = os.path.join(basedir, f"{j:04d}.npy")
                try:
                    vertices[i] = np.load(exp_verts_npy_dir)
                except:
                    raise ValueError(f'something is wrong! {exp_verts_npy_dir}')
            # for alignment
            v0 = np.load(os.path.join(basedir, f"{0:04d}.npy"))
        else:
            exp_verts_file = basedir+'.npy'
            vertices = np.load(exp_verts_file)
            slice_idx = self.get_slice_idx(vertices, WS)
            
            # for alignment
            v0 = vertices[0]
            print(f'saving each frame at: {basedir}')
            os.makedirs(basedir, exist_ok=True)
            for j, verts_npy in enumerate(vertices):
                exp_verts_npy_dir = os.path.join(basedir, f"{j:04d}")
                np.save(exp_verts_npy_dir, verts_npy)
            vertices = vertices[slice_idx:slice_idx+WS]
        
        return vertices, v0
    
    def augment_trans_scale(self, template, vertices):
        ## Random Augmentation ---------------------------------------------------------
        trans, scale = 0.0, 1.0
        if self.opts.data_rand_trans:
            trans = (torch.rand((1, 3)) - 0.5) * 0.5
        if self.opts.data_rand_scale:
            scale = torch.rand((1)).repeat(3) * 0.4 + 0.8
        template = template * scale + trans
        vertices = vertices * scale + trans
        ## -----------------------------------------------------------------------------
        
        return template, vertices
    
    def __getitem__(self, index):
        """
        Todo:
        - [ ] add FLAME - distill neckpose
        - [ ] adding LYHM, COMA, FaMoS, D3DFACS
            - LYHM registrations (1216 subjects, one neutral expression scan each)
            - CoMA registrations (12 subjects, 12 extreme dynamic expressions each)
            - D3DFACS registrations (10 subjects, 519 dynamic expressions in total)
            - FaMoS registrations (95 subjects, 28 dynamic expressions each, 600K frames in total)
        """
        
        for len_data, get_data, mesh_data in self.len_list:
            if index < len_data:
                datas = get_data(index)
                break
            else:
                index = index - len_data
        
        # audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img, audio_path = get_data(index)
        audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, v_normal, faces, img, audio_path = datas
        # self.identity_num[audio_path.split('/')[5]]
        
        ## Random Augmentation ---------------------------------------------------------
        template, vertices = self.augment_trans_scale(template, vertices)
        ## -----------------------------------------------------------------------------
        
        torch.cuda.empty_cache()
        
        if self.return_audio_dir:
            return audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, v_normal, faces, img, mesh_data, audio_path
        else:
            return audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, v_normal, faces, img, mesh_data
        # audio_feat, id_coeff, exp_coeff, template, dfn_info, vertices, faces, img = self.get_ICTcapture(index)
    
    def get_slice_idx(self, F_idx, WS):
        """
        Args:
            F_idx (int / np.ndarray): frame index for exp_coeff or vertices
        Returns:
            slice_idx (int): slicing index
        """
        if type(F_idx)==np.ndarray:
            F_idx = F_idx.shape[0]
        if type(F_idx)==torch.Tensor:
            F_idx = F_idx.shape[0]
            
        if self.mode == 'test':
            WS = F_idx
            slice_idx = 0
        else:
            slice_idx = random.randint(0, F_idx-WS)
        return slice_idx
    
    def get_id_num(self, audio_path, template):
        if template.shape[0] == self.ict_face_model.v_idx:
            return self.identity_num[audio_path.split(self.mode)[-1].split('/')[1]]
        
        # elif self.use_voca and template.shape[0] == self.voca_trimesh.vertices.shape[0]:        
        elif self.use_voca and template.shape[0] == self.voca_std['v_idx'].shape[0]:
            return self.identity_num[audio_path.split(self.mode)[-1].split('/')[1]]
        
        elif self.use_biwi and template.shape[0] == self.biwi_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('self.mode')[-1].split('/')[-1].split('_')[0]]
        
        # elif self.use_mf_ROM and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
        #     return
        
        elif self.use_mf_SEN and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('wav2vec2')[-1].split('/')[-1].split('-SEN')[0]]
    
    def vis_mesh(self, vertices, frame=0, mode='ict', tag=''):
        if mode == 'ict':
            faces = self.ict_face_model.faces
        elif mode == 'voca':
            #faces = self.voca_trimesh.faces
            faces = self.voca_std['new_f']
        elif mode == 'biwi':
            faces = self.biwi_trimesh.faces
        elif mode == 'mf':
            faces = self.mf_std['new_f']
            
        rot_list=[[0,90,0], [0,0,0], [0,-90,0]]
        len_rot = len(rot_list)
        v_list  = [vertices[frame]]*len_rot
        f_list  = [faces]*len_rot
        plot_image_array(
            v_list, f_list, rot_list, 
            size=3,
            mode='shade', 
            bg_black=False, 
            logdir='_tmp',
            save=True,
            name=f'{tag}-{mode}_{frame:03d}'
        )

class InvRigDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 ict_basedir: str = '/data/ICT-audio2face/split_set/', 
                 is_train: bool = False,
                 is_valid: bool = False,
                 window_size: int = 8, # batch size
                 ict_face_only: bool = True,
                 device: str = 'cpu',
                 print_config: bool= False,
                ):
        super(InvRigDataset, self).__init__()
        """
        ICT only data
        """
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.WS = self.opts.window_size if self.opts is not None else window_size
        self.device = device
        
        # not used
        self.ict_face_only = self.opts.ict_face_only if self.opts is not None else ict_face_only
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
        
        ## ICT-facekit --------------------------------------------------------------------
        self.iden_vecs = torch.from_numpy(
            np.load('./data/ICT_live_100/iden_vecs.npy')
        ).float() #[:8]
        self.expression_vecs = torch.from_numpy(
            np.load(f'./data/ICT_live_100/expression_vecs_{self.mode}.npy')
        ).float()[:128]

        self.ict_face_model = ICT_face_model(face_only=False)
        self.ict_precompute_path = f'./ICT/precompute-fullhead'
        self.ict_precompute_path_synth = f'./ICT/precompute-synth-fullhead'

        self.ict_face_model_fo = ICT_face_model(face_only=True)
        self.ict_precompute_path_fo = './ICT/precompute-face_only'
        self.ict_precompute_path_fo_synth = './ICT/precompute-synth-face_only'
        
        self.ict_basedir = os.path.join(ict_basedir,  self.mode)
        self.ict_vert_segment = torch.from_numpy(np.load('./utils/ict/ICT_segment_onehot.npy'))
        ## --------------------------------------------------------------------------------
        
        ## --------------------------------------------------------------------------------
        self.total_len = self.iden_vecs.shape[0] * self.expression_vecs.shape[0]

        if print_config:
            print(self.get_data_config())
    
    def get_data_config(self):
        text = "========[ICT_InvRigDataset]========\n"
        text += f"mode: {self.mode}\n"
        text += f"id: {self.iden_vecs.shape[0]}\n"
        text += f"exp: {self.expression_vecs.shape[0]}\n"
        text += f"-----------------------\n"
        text += f"[total data]: {self.total_len}\n"
        text += "===============================\n"
        return text
    
    def __len__(self):
        return self.total_len
        
    def get_ICTsynthetic(self, index):
        e_index = index % self.expression_vecs.shape[0]
        if self.expression_vecs.shape[0] - e_index < self.WS:
            e_index = self.expression_vecs.shape[0] - self.WS
        id_idx = np.random.randint(0, self.iden_vecs.shape[0])
        
        # get blendshape coeffs
        id_coeff = torch.cat([self.iden_vecs[id_idx], torch.zeros(28)]) # ------- [128]
        exp_coeff = self.expression_vecs[e_index:e_index+self.WS] # -------------- [W, 53]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(self.WS, 75)], dim=-1) # --- [W, 128]
        
        # get template vertices and face indices (neutral face)
        if not self.ict_face_only:
            if random.random() > 0.5:
                ict_model = self.ict_face_model
                ict_path_synth = self.ict_precompute_path_synth
            else:
                ict_model = self.ict_face_model_fo
                ict_path_synth = self.ict_precompute_path_fo_synth
        
        vertices, template, _ = ict_model.apply_coeffs(
            id_coeff[:100].numpy(), 
            exp_coeff[:,:53].numpy(), 
            return_all=True
        )
        vertices = torch.from_numpy(vertices).float() # -------------------------- [W, V, 3]
        template = torch.from_numpy(template[0]).float() # ----------------------- [V, 3]
        faces = torch.from_numpy(ict_model.faces).long() # ----------------------- [F, 3]
        
        # get template dfn_info (neutral face)
        dfn_info  = pickle.load(open(os.path.join(
            ict_path_synth,
            f"{id_idx:03d}_dfn_info.pkl"
        ), 'rb'))

        operators = os.path.join(
            ict_path_synth, 
            f"{id_idx:03d}_operators.pkl"
        )
        
        img = np.load(os.path.join(ict_path_synth, f"{id_idx:03d}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        dummy = torch.zeros(self.WS, 768)
        
        v_normal = calc_norm_torch(vertices, faces, at='v')

        template, vertices = self.augment_trans_scale(template, vertices)

        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img, ''
        
    def augment_trans_scale(self, template, vertices):
        trans, scale = 0.0, 1.0
        if self.opts.data_rand_trans:
            trans = (torch.rand((1, 3)) - 0.5) * 0.5
        if self.opts.data_rand_scale:
            scale = torch.rand((1)).repeat(3) * 0.4 + 0.8
        template = template * scale + trans
        vertices = vertices * scale + trans
        
        return template, vertices

    def __getitem__(self, index):
        return self.get_ICTsynthetic(index)

class MeshDataBatch:
    def __init__(self, data):
        """
        Args:
            audio_feat
            id_coeff
            gt_rig_params
            template
            dfn_info
            operators
            vertices
            faces
            img
            mesh_data
            audio_path
        """
        if data is not None: # essential !
            transposed_data = list(zip(*data))
            
            self.audio_feat = torch.stack(transposed_data[0], 0)
            self.id_coeff = torch.stack(transposed_data[1], 0)
            self.gt_rig_params = torch.stack(transposed_data[2], 0)
            
            self.vertices = torch.stack(transposed_data[6], 0)
            self.normals = torch.stack(transposed_data[7], 0)
            
            if len(transposed_data) > 11:
                self.audio_path = transposed_data[11]
            
            # pickle path, same mesh in minibatch == same operator!
            self.mesh_data = transposed_data[10][0]
            self.template = transposed_data[3][0][None]
            self.img = transposed_data[9][0]
            self.faces = transposed_data[8][0]
            self.operators = transposed_data[5][0]
            self.set_dfn_info(transposed_data[4][0])
            
            # transposed_dfn_info = list(zip(*transposed_data[4]))
            # self.set_batch_dfn_info(transposed_dfn_info)
    
    @property
    def get_dfn_info(self): 
        return [self.mass, self.L, self.evals, self.evecs, self.grad_X, self.grad_Y, self.faces]
        
    def set_dfn_info(self, data):
        # DiffusionNet precomputes
        self.mass = data[0]
        self.L = data[1]
        self.evals = data[2]
        self.evecs = data[3]
        self.grad_X = data[4]
        self.grad_Y = data[5]
        #self.faces = torch.stack(dfn_info[6], 0) # -> duplicated!
        
    def set_batch_dfn_info(self, dfn_info):
        # DiffusionNet precomputes
        self.mass = torch.stack(dfn_info[0], 0)
        self.L = torch.stack(dfn_info[1], 0)
        self.evals = torch.stack(dfn_info[2], 0)
        self.evecs = torch.stack(dfn_info[3], 0)
        self.grad_X = torch.stack(dfn_info[4], 0)
        self.grad_Y = torch.stack(dfn_info[5], 0)
        #self.faces = torch.stack(dfn_info[6], 0) # -> duplicated!
    
    def to(self, device='cpu'):
        for id_ in self.__dict__.keys():
            attr = self.__getattribute__(id_)
            if isinstance(attr, torch.Tensor):
                self.__setattr__(id_, attr.to(device))
        return self
        
    # # custom memory pinning method on custom type
    # def pin_memory(self):
    #     self.inp = self.inp.pin_memory()
    #     self.tgt = self.tgt.pin_memory()
    #     return self
    
def collate_wrapper(batch, device="cpu"):
    return MeshDataBatch(batch).to(device)

if __name__ == "__main__":
    """
    python dataloader_mesh.py
    """
    from utils import plot_image_array, plot_image_array_diff3, vis_rig 
    import yaml; import argparse
    from tqdm import tqdm
    opts_yaml = yaml.load(open('config/train.yml'), Loader=yaml.FullLoader)
    opts_yaml["learn_rig_emb"] = False
    opts_yaml["device"] = "cuda:0"
    opts = argparse.Namespace(**opts_yaml)
    
    # opts.selection = 4 # BIWI
    # opts.selection = 3 # VOCASET
    opts.selection = 20 # ict and multiface (mf)
    # opts.selection = 1 # ict-capture
    # opts.selection = 2
    
    # opts.selection = 5 # mf SEN
    # opts.selection = 6 # mf ROM
    
    opts.window_size = 8
    print(f'use window_size: {opts.window_size}')
    
    opts.batch_size = 1
    print(f'use batch_size: {opts.batch_size}')
    
    #dataset = InvRigDataset(opts, is_train=True)
    dataset = NFSDataset(opts, is_train=True, return_audio_dir=True)
    print(dataset.get_data_config())
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=False,
        collate_fn=partial(collate_wrapper, device=opts.device), num_workers=0)
    len_dataloader = len(dataloader)
    #data = next(iter(dataloader))
    # iter_dataloader = iter(dataloader)
    # pbar = tqdm(enumerate(dataloader), total=len_dataloader, position=0)
    # rambar = tqdm(range(len_dataloader), total=125, desc='ram', position=1, ncols=100)
    # cpubar = tqdm(range(len_dataloader), total=100, desc='cpu', position=2, ncols=100)
    
    for idx, batch in enumerate(dataloader):
        #(audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data, audio_path = batch
        #print(audio_path[0], audio_feat.shape, id_coeff.shape, gt_rig_params.shape, template.shape, vertices.shape)
        mode = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data.cpu().numpy()]
        print(idx, mode, batch.audio_feat.shape, batch.gt_rig_params.shape, batch.template.shape, batch.vertices.shape, batch.normals.shape)
        #print(template.min(), template.max())

#         if batch.template.shape[1]==dataset.ict_face_model.v_idx:
#             mode= 'ict'
#         elif batch.template.shape[1]==3525:
#             mode='voca'
#         elif batch.template.shape[1]==23370:
#             mode='biwi'
#         elif batch.template.shape[1]==5223:
#             mode='mf'
#         else:
#             mode='mf'

        #dataset.vis_mesh(dataset.biwi_trimesh.vertices[None], frame=0, mode=mode)
        #T = 0
        # dataset.vis_mesh(vertices[0, T:T+1], frame=0, mode=mode, tag=f'{idx:04d}-anim')
        # dataset.vis_mesh(template, frame=0, mode=mode, tag=f'{idx:04d}-temp')
        # vis_rig(gt_rig_params, save_fn=f'_tmp/{idx:04d}-rig_vis.jpg', normalize=True)
        
        #self.vis_mesh(self.ict_template, frame=0, mode='ict', tag=f'testict-rig_vis')
    
    ## configure alignment ----------------
    #dataset.voca_trimesh.vertices = template[0]; dataset.voca_trimesh.faces = faces[0]; _=dataset.voca_trimesh.export('voca_template.obj', file_type='obj')
    #dataset.biwi_trimesh.vertices = template[0]; dataset.biwi_trimesh.faces = faces[0]; _=dataset.biwi_trimesh.export('biwi_template.obj', file_type='obj')
    
    dataset.vis_mesh(batch.vertices[0,:1], frame=0, mode=mode)
    #-----------------------------------------------------------------------------------
    
#     audio_feat_types = ['wav2vec2', 'hubert']
#     audio_feat_levels = []
#     audio_feat_levels += [['logits'] + [f"{i:02d}" for i in range(0,12)]]
#     audio_feat_levels += [['logits'] + [f"{i:02d}" for i in range(0,24)]]

#     for audio_feat_type, audio_feat_level in zip(audio_feat_types, audio_feat_levels):
#         for index, feat_level in enumerate(audio_feat_level):
#             print("=========================================")
#             print(f"audio_feat_type: {audio_feat_type}, audio_feat_level: {feat_level}")
#             dataset = Wav2RigDatasetDistill(
#                                      basedir='/data/ICT-audio2face/data_30fps', 
#                                      audio_feat_type=audio_feat_type,
#                                      audio_feat_level=feat_level,
#                                      is_train=True,
#                                      return_one_hot_id=True,
#                                      return_audio_dir=True,
#                                     )
#             dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#             for index, batch in enumerate(dataloader):
#                 anim, audio, audio_fn, one_hot, z_distill = batch
#                 if index % 20 == 0:
#                     print(anim.shape, z_distill.shape)
#                     print("audio_fn:", audio_fn[0], one_hot[0])
#                 if anim.shape[1] != z_distill.shape[1]:
#                     print("anim.shape[1] != z_distill.shape[1]")
#                     print(anim.shape[1],z_distill.shape[1])

#                 if anim.shape[1] != audio.shape[1]:
#                     print("anim.shape[1] != audio.shape[1]")
#                     print(anim.shape[1],audio.shape[1])

#             dataset = Wav2RigDatasetDistill(basedir='/data/ICT-audio2face/data_30fps',
#                                         audio_feat_type=audio_feat_type,
#                                         audio_feat_level=feat_level,
#                                         is_train=False,
#                                         return_one_hot_id=True,
#                                         return_audio_dir=True,
#                                         )
#             dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
#             for batch in dataloader:
#                 anim, audio, audio_fn, one_hot, z_distill = batch

#                 print(anim.shape, z_distill.shape)
#                 print("audio_fn:", audio_fn[0], one_hot[0])
#                 if anim.shape[1] != audio.shape[1]:
#                     print("anim.shape[1] != audio.shape[1]")
#                     print(anim.shape[1],audio.shape[1])
#             exit()