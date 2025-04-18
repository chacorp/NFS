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
from tqdm import tqdm

import time

def random_ict_exp_coeff(N=1000, min_ones=1, max_ones=5):
    identity = np.eye(53)

    num_ones_per_row = np.random.randint(min_ones, max_ones + 1, size=N)

    exp_coeffs = np.array([np.sum(identity[np.random.choice(53, num_ones, replace=False)], axis=0)
                    for num_ones in num_ones_per_row], dtype=int)
    return exp_coeffs

class MeshDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 data_basedir='/data/sihun/',
                 is_train=False,
                 is_valid=False,
                 window_size=8, # batch size
                 device='cpu',
                 print_config=False,
                 ict_face_only=False
                ):
        super().__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.device = device
        self.data_basedir = data_basedir
        self.ict_face_only=ict_face_only
        self.data_config(self.opts, window_size)
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
            
        ## precomputes ------------------------------------------------------------------
        p_mode = 'face_only' if self.ict_face_only else 'fullhead'
        self.ict_data_split, \
        self.voca_data_split, \
        self.biwi_data_split, \
        self.mf_data_split, \
        self.ict_data_synth = get_data_splits()
        
        self.identity_num = get_identity_num()
                
        ## loading data -----------------------------------------------------------------
        # self.set_multiface_SEN()
        # self.set_multiface_ROM()
        # self.set_biwi()
        self.set_ict_base()
        ## ------------------------------------------------------------------------------
        
        self.len_ict_synth = 0
        self.len_ict_real = 0
        self.len_mfROM = 0
        self.len_mfSEN = 0
        self.total_len = 0
        
        self.total_len = 0
        self.id_sent_len = 0
        self.len_list = []
        if self.use_ict_real:
            total_len, id_sent_len = self.set_ict_real()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([total_len, self.get_ICTcapture, torch.tensor(0), id_sent_len])
        if self.use_ict_synth:
            total_len, id_sent_len = self.set_ict_synth()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([self.total_len, self.get_ICTsynthetic, torch.tensor(0), id_sent_len])
        if self.use_ict_synth_single:
            total_len, id_sent_len = self.set_ict_synth_single()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([self.total_len, self.get_ICTsynthetic, torch.tensor(0), id_sent_len])
        if self.use_voca:
            total_len, id_sent_len = self.set_voca()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([self.total_len, self.get_voca, torch.tensor(1), id_sent_len])
        if self.use_coma:
            total_len, id_sent_len = self.set_coma()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([self.total_len, self.get_coma, torch.tensor(1), id_sent_len])
        if self.use_biwi:
            total_len, id_sent_len = self.set_biwi()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([self.total_len, self.get_biwi, torch.tensor(2), id_sent_len])
        if self.use_mf_SEN:
            total_len, id_sent_len = self.set_multiface_SEN()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([total_len, self.get_multiface_SEN, torch.tensor(3), id_sent_len])
        if self.use_mf_ROM:
            total_len, id_sent_len = self.set_multiface_ROM()
            self.total_len += total_len
            self.id_sent_len += id_sent_len
            self.len_list.append([total_len, self.get_multiface_ROM, torch.tensor(3), id_sent_len])
        #self.total_len = self.len_ict_real+self.len_ict_synth+self.len_mfSEN+self.len_mfROM
        
        if print_config:
            print(self.get_data_config())
        
    def get_data_config(self):
        text = "===========[Dataset]===========\n"
        text+= f"[mode]: {self.mode}\n"
        # text+= f"[ICT-synth data]: {self.len_ict_synth}\n"
        # text+= f"[ICT-capture data]: {self.len_ict_real}\n"
        # text+= f"[multiface data]: {self.len_mfSEN+self.len_mfROM}\n"
        text+= f"------------------------------\n"
        text+= f"[total frame]: {self.total_len}\n"
        text+= f"[data chunks (ID x Motion) - matching batch_size]: {self.id_sent_len}\n"
        text+= "===============================\n"
        return text
    
    def __len__(self):
        #return self.total_len
        return self.id_sent_len
    
    def data_config(self, opts=None, window_size=8):
        flag = True if opts is not None else False
        
        self.use_ict_synth_single = opts.use_ict_synth_single if flag else False
        
        self.use_ict_real = opts.use_ict_real if flag else False
        self.use_ict_synth = opts.use_ict_synth if flag else False
        self.use_mf_SEN = opts.use_mf_SEN if flag else False
        self.use_mf_ROM = opts.use_mf_ROM if flag else False
        self.use_voca = opts.use_voca if flag else False
        self.use_coma = opts.use_coma if flag else False
        self.use_biwi = opts.use_biwi if flag else False
        
        if flag:
            self.use_decimate = False
            self.WS = window_size
        else:
            self.use_decimate = self.opts.use_decimate
            self.WS = self.opts.window_size
            
    def set_multiface_SEN(self):
        self.mf_dir = os.path.join(self.data_basedir, 'multiface_align')
        self.mf_precompute_path=f'{self.mf_dir}/precomputes'
        self.mf_obj_path=f'{self.mf_dir}/obj'
        self.mf_std = np.load('utils/mf/standardization.npy', allow_pickle=True).item()
        self.mf_verts = np.load(f"{self.mf_dir}/id_verts_{self.mode}.npy")
        
        self.mf_SEN_dir = f'{self.mf_dir}/SEN/{self.mode}'
        self.mf_SEN_v_npy_list = {}
        self.mf_SEN_n_npy_list = {}
        self.mf_SEN_wav_list = {} # TODO
        
        self.mf_SEN_count = [0] # start
        self.mf_SEN_remain = [0] # start
        self.mf_id = []
        
        total_len=0
        id_sent_len = 0
        for id_ in tqdm(self.mf_data_split[self.mode], desc='multiface SEN'):
            self.mf_SEN_v_npy_list[id_]={}
            self.mf_SEN_n_npy_list[id_]={}
            
            curr_v_dir = f"{self.mf_SEN_dir}/vertices_npy/{id_}"
            v_listdirs = sorted([f for f in os.listdir(curr_v_dir) if f!='.ipynb_checkpoints'])
            
            curr_n_dir = f"{self.mf_SEN_dir}/normals_npy/{id_}"
            n_listdirs = sorted([f for f in os.listdir(curr_n_dir) if f!='.ipynb_checkpoints'])
            
            assert len(v_listdirs) == len(n_listdirs), 'sentence lenth do not match'
            
            for sen_ in v_listdirs:
                self.mf_SEN_v_npy_list[id_][sen_] = sorted(glob.glob(f"{curr_v_dir}/{sen_}/*.npy"))
                self.mf_SEN_n_npy_list[id_][sen_] = sorted(glob.glob(f"{curr_n_dir}/{sen_}/*.npy"))
                
                assert len(self.mf_SEN_v_npy_list[id_][sen_]) == len(self.mf_SEN_n_npy_list[id_][sen_]), 'mismatch'
                
                total_len += len(self.mf_SEN_v_npy_list[id_][sen_])
            remain = len(v_listdirs) % self.opts.batch_size

            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.mf_SEN_count.append(n_id_sent)
            self.mf_SEN_remain.append(dummy)
        
        self.mf_SEN_count=np.array(self.mf_SEN_count)
        self.mf_SEN_count=np.cumsum(self.mf_SEN_count)
        self.mf_SEN_remain=np.array(self.mf_SEN_remain)

        return total_len, id_sent_len
    
    def set_multiface_ROM(self):
        self.mf_dir = os.path.join(self.data_basedir, 'multiface_align')
        self.mf_precompute_path=f'{self.mf_dir}/precomputes'
        self.mf_obj_path=f'{self.mf_dir}/obj'
        self.mf_std = np.load('utils/mf/standardization.npy', allow_pickle=True).item()
        self.mf_verts = np.load(f"{self.mf_dir}/id_verts_{self.mode}.npy")
        
        self.mf_ROM_dir = f'{self.mf_dir}/ROM/{self.mode}'
        self.mf_ROM_v_npy_list = {}
        self.mf_ROM_n_npy_list = {}
        self.mf_ROM_count = [0] # start
        self.mf_ROM_remain = [0] # start
        self.mf_id = []

        total_len=0
        id_sent_len = 0
        for idx, id_ in tqdm(enumerate(self.mf_data_split[self.mode]), desc='multiface ROM'):
            self.mf_ROM_v_npy_list[id_]={}
            self.mf_ROM_n_npy_list[id_]={}
            
            curr_v_dir = f"{self.mf_ROM_dir}/vertices_npy/{id_}"
            v_listdirs = sorted([f for f in os.listdir(curr_v_dir) if f!='.ipynb_checkpoints'])
            
            curr_n_dir = f"{self.mf_ROM_dir}/normals_npy/{id_}"
            n_listdirs = sorted([f for f in os.listdir(curr_n_dir) if f!='.ipynb_checkpoints'])
            
            assert len(v_listdirs) == len(n_listdirs), 'sentence lenth do not match'
            
            self.mf_id.append(idx)
            for sen_ in v_listdirs:
                v_npy_list = sorted(glob.glob(f"{self.mf_ROM_dir}/vertices_npy/{id_}/{sen_}/*.npy"))
                n_npy_list = sorted(glob.glob(f"{self.mf_ROM_dir}/normals_npy/{id_}/{sen_}/*.npy" ))

                assert len(v_npy_list) == len(n_npy_list)
                
                self.mf_ROM_v_npy_list[id_][sen_] = v_npy_list
                self.mf_ROM_n_npy_list[id_][sen_] = n_npy_list

                total_len += len(self.mf_ROM_v_npy_list[id_][sen_]) # SEN frame len

            remain = len(v_listdirs) % self.opts.batch_size
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.mf_ROM_count.append(n_id_sent)
            self.mf_ROM_remain.append(dummy)

        self.mf_ROM_count=np.array(self.mf_ROM_count)
        self.mf_ROM_count=np.cumsum(self.mf_ROM_count)
        self.mf_ROM_remain=np.array(self.mf_ROM_remain)

        return total_len, id_sent_len
    
    def set_biwi(self):
        biwi_base_dir = os.path.join(self.data_basedir, 'BIWI_align_deci')
        self.biwi_precompute_path = f'{biwi_base_dir}/precomputes'
        self.biwi_templates = pickle.load(open(f'{biwi_base_dir}/templates_align_deci.pkl', 'rb'), encoding='latin1')
        self.biwi_std = np.load('./utils/biwi/standardization.npy', allow_pickle=True).item() # not used
        
        self.biwi_dir = f'{biwi_base_dir}/{self.mode}'
        self.biwi_v_npy_list = {}
        self.biwi_n_npy_list = {}
        self.biwi_wav_list = {} # TODO
        #self.biwi_len_list = {}
        
        self.biwi_count = [0] # start
        self.biwi_remain = [0] # start
        
        self.biwi_id = []
        
        total_len = 0
        id_sent_len = 0
        for id_ in tqdm(self.biwi_data_split[self.mode], desc='BIWI'):
            
            self.biwi_v_npy_list[id_]={}
            self.biwi_n_npy_list[id_]={}
                
            v_listdirs = [f.split('/')[-1] for f in glob.glob(f"{self.biwi_dir}/vertices_npy/{id_}*")]
            n_listdirs = [f.split('/')[-1] for f in glob.glob(f"{self.biwi_dir}/normals_npy/{id_}*")]
            
            assert len(v_listdirs) == len(n_listdirs), 'sentence lenth do not match'
            
            self.biwi_id.append(id_)
            for id_sent in v_listdirs:
                v_npy_list = sorted(glob.glob(f"{self.biwi_dir}/vertices_npy/{id_sent}/*.npy"))
                n_npy_list = sorted(glob.glob(f"{self.biwi_dir}/normals_npy/{id_sent}/*.npy" ))
                
                assert len(v_npy_list) == len(n_npy_list)
                
                id_, sent = id_sent.split('_')
                self.biwi_v_npy_list[id_][sent] = v_npy_list
                self.biwi_n_npy_list[id_][sent] = n_npy_list
                
                total_len += len(self.biwi_v_npy_list[id_][sent]) # SEN frame len
                
            remain = len(v_listdirs) % self.opts.batch_size
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.biwi_count.append(n_id_sent)
            self.biwi_remain.append(dummy)
        
        self.biwi_count=np.array(self.biwi_count)
        self.biwi_count=np.cumsum(self.biwi_count)
        self.biwi_remain=np.array(self.biwi_remain)
        
        return total_len, id_sent_len
    
    def set_voca(self):
        self.all_dir = os.path.join(self.data_basedir, 'VOCA-COMA')
        self.voca_dir = f'{self.all_dir}/VOCASET/{self.mode}'
        self.voca_coma_precompute_path=f'{self.all_dir}/precomputes'
        self.voca_coma_templates = pickle.load(open(f'{self.all_dir}/voca_templates.pkl', 'rb'), encoding='latin1')
        #dfn_info = pickle.load(open(self.mf_dir, 'rb'))
        
        self.voca_coma_std = np.load('./utils/voca/standardization.npy', allow_pickle=True).item()
        
        self.voca_v_npy_list = {}
        self.voca_n_npy_list = {}
        self.voca_wav_list = {} # TODO
        #self.voca_len_list = {}
        
        self.voca_count = [0] # start
        self.voca_remain = [0] # start
        
        total_len = 0
        id_sent_len = 0
        for id_ in tqdm(self.voca_data_split[self.mode], desc='VOCA'):
            self.voca_v_npy_list[id_]={}
            self.voca_n_npy_list[id_]={}
            
            curr_v_dir = f"{self.voca_dir}/{id_}/vertices_npy"
            v_listdirs = sorted([f for f in os.listdir(curr_v_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            curr_n_dir = f"{self.voca_dir}/{id_}/normals_npy"
            n_listdirs = sorted([f for f in os.listdir(curr_n_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            assert v_listdirs == n_listdirs, 'sentence lenth do not match'
            
            for sen_ in v_listdirs:
                self.voca_v_npy_list[id_][sen_]=sorted(glob.glob(f"{curr_v_dir}/{sen_}/*30fps*.npy"))
                self.voca_n_npy_list[id_][sen_]=sorted(glob.glob(f"{curr_n_dir}/{sen_}/*30fps*.npy"))
                
                assert len(self.voca_v_npy_list[id_][sen_]) == len(self.voca_n_npy_list[id_][sen_])
                
                total_len+=len(self.voca_v_npy_list[id_][sen_])
                
            remain = len(v_listdirs) % self.opts.batch_size
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.voca_count.append(n_id_sent)
            self.voca_remain.append(dummy)
            
        self.voca_count=np.array(self.voca_count)
        self.voca_count=np.cumsum(self.voca_count)
        self.voca_remain=np.array(self.voca_remain)
        
        return total_len, id_sent_len
    
    def set_coma(self):
        self.all_dir = os.path.join(self.data_basedir, 'VOCA-COMA')
        self.coma_dir = f'{self.all_dir}/COMA/{self.mode}'
        self.voca_coma_precompute_path=f'{self.all_dir}/precomputes'
        self.voca_coma_templates = pickle.load(open(f'{self.all_dir}/voca_templates.pkl', 'rb'), encoding='latin1')
        #dfn_info = pickle.load(open(self.mf_dir, 'rb'))
        
        self.voca_coma_std = np.load('./utils/voca/standardization.npy', allow_pickle=True).item()
        
        self.coma_dir = f'{self.all_dir}/COMA/{self.mode}'
        self.coma_v_npy_list = {}
        self.coma_n_npy_list = {}
        # self.coma_len_list = {}
        
        self.coma_count = [0] # start
        self.coma_remain = [0] # start
        
        total_len=0
        id_sent_len=0
        # coma has same identities with voca
        #import pdb;pdb.set_trace()
        for id_ in tqdm(self.voca_data_split[self.mode], desc='COMA'):
            self.coma_v_npy_list[id_]={}
            self.coma_n_npy_list[id_]={}
            
            curr_v_dir = f"{self.coma_dir}/{id_}/vertices_npy"
            v_listdirs = sorted([f for f in os.listdir(curr_v_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            curr_n_dir = f"{self.coma_dir}/{id_}/normals_npy"
            n_listdirs = sorted([f for f in os.listdir(curr_n_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            assert v_listdirs == n_listdirs, 'sentence lenth do not match'
            
            for sen_ in v_listdirs:
                self.coma_v_npy_list[id_][sen_]=sorted(glob.glob(f"{curr_v_dir}/{sen_}/*.npy"))
                self.coma_n_npy_list[id_][sen_]=sorted(glob.glob(f"{curr_n_dir}/{sen_}/*.npy"))
                
                assert len(self.coma_v_npy_list[id_][sen_]) == len(self.coma_n_npy_list[id_][sen_])
                
                total_len+=len(self.coma_v_npy_list[id_][sen_])
            
            remain = len(v_listdirs) % self.opts.batch_size
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.coma_count.append(n_id_sent)
            self.coma_remain.append(dummy)
            
        self.coma_count=np.array(self.coma_count)
        self.coma_count=np.cumsum(self.coma_count)
        self.coma_remain=np.array(self.coma_remain)
        
        return total_len, id_sent_len
    
    def set_ict_base(self):
        self.ict_basedir = os.path.join(self.data_basedir, 'ICT-audio2face')
        self.ict_templates_path = './ICT/templates'
        
        with open("/data/sihun/ICT-audio2face/split_set/ict_real_templates.pkl", 'rb') as f:
            self.ict_real_templates_dict = pickle.load(f)
            
        with open("/data/sihun/ICT-audio2face/synth_set/ict_synth_templates.pkl", 'rb') as f:
            self.ict_synth_templates_dict = pickle.load(f)
        self.ict_face_model = ICT_face_model()
        
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
        
    def set_ict_real(self):
        #self.iden_vecs = np.load('./ict_face_pt/random_identity_vecs.npy')
        self.ict_real_id_vecs = np.load(f'{self.ict_basedir}/split_set/id_vecs.npy')
        
        self.ict_real_precompute = f'{self.ict_basedir}/precompute-real-fullhead'
        self.ict_real_precompute_fo = f'{self.ict_basedir}/precompute-real-face_only'
        
        self.ict_dir = os.path.join(self.ict_basedir, 'split_set', self.mode)
        
        self.ict_real_count = [0] # start
        self.ict_real_remain = [0] # start
        
        self.ict_real_v_npy_list = {}
        self.ict_real_n_npy_list = {}
        self.ict_real_exp_coeff_list = {}
        self.ict_real_wav_list = {} # TODO
        #self.ict_real_len_list = {}
        
        total_len=0
        id_sent_len=0
        for id_ in tqdm(self.ict_data_split[self.mode], desc='ict real'):
            self.ict_real_v_npy_list[id_]={}
            self.ict_real_n_npy_list[id_]={}
            
            curr_v_dir = f"{self.ict_dir}/{id_}/vertices_npy"
            v_listdirs = sorted([f for f in os.listdir(curr_v_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            curr_n_dir = f"{self.ict_dir}/{id_}/normals_npy"
            n_listdirs = sorted([f for f in os.listdir(curr_n_dir) if f!='.ipynb_checkpoints' and not '.npy' in f])
            
            assert v_listdirs == n_listdirs, 'sentence lenth do not match'
            
            curr_r_dir = f"{self.ict_dir}/{id_}/rig_param"
            self.ict_real_exp_coeff_list[id_]=sorted(glob.glob(f"{curr_r_dir}/*.npy"))
            
            for sen_ in v_listdirs:
                v_npy_list = sorted(glob.glob(f"{curr_v_dir}/{sen_}/*.npy"))
                n_npy_list = sorted(glob.glob(f"{curr_n_dir}/{sen_}/*.npy"))
                
                assert len(v_npy_list) == len(n_npy_list), f"{id_}-{sen_}"
                
                self.ict_real_v_npy_list[id_][sen_]=v_npy_list
                self.ict_real_n_npy_list[id_][sen_]=n_npy_list
                
                # self.ict_real_wav_list[id_]=f"{self.ict_dir}/{id_}/wav/{sen_}.wav" ## TODO

                total_len += len(self.ict_real_v_npy_list[id_][sen_])
            #id_sent_len += len(self.ict_real_v_npy_list[id_])
            
            remain = len(v_listdirs) % self.opts.batch_size
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.ict_real_count.append(n_id_sent)
            self.ict_real_remain.append(dummy)
            
            assert len(self.ict_real_exp_coeff_list[id_]) == len(self.ict_real_v_npy_list[id_]), \
                f"exp_coeff and v mismatch in {id_}"
        
        #id_key = self.ict_data_split[self.mode][id_idx]
        self.ict_sent_key_list = list(self.ict_real_v_npy_list['m00'].keys())
        self.ict_sent_len = len(self.ict_real_v_npy_list['m00'].keys())
        
        self.ict_real_count=np.array(self.ict_real_count)
        self.ict_real_count=np.cumsum(self.ict_real_count)
        self.ict_real_remain=np.array(self.ict_real_remain)
        
        return total_len, id_sent_len
          
    def set_ict_synth(self):
        
        self.iden_vecs = np.load('./ict_face_pt/random_identity_vecs.npy')
        self.expression_vecs = np.load('./ict_face_pt/random_expression_vecs.npy')
        
        if self.mode != 'train':
            self.iden_vecs = np.load('./data/ICT_live_100/iden_vecs.npy')
            self.expression_vecs = np.load(f'./data/ICT_live_100/expression_vecs_{self.mode}.npy')
        
        self.ict_synth_precompute = f'{self.ict_basedir}/precompute-synth-fullhead'
        self.ict_synth_precompute_fo = f'{self.ict_basedir}/precompute-synth-face_only'
        self.ict_synth_precompute_nf = f'{self.ict_basedir}/precompute-synth-narrow_face'
        
        self.ict_dir = os.path.join(self.ict_basedir, 'synth_set', self.mode)
        
        self.ict_synth_v_npy_list = {}
        self.ict_synth_n_npy_list = {}
        #self.ict_synth_len_list = {}
        
        self.ict_synth_count = [0]
        self.ict_synth_remain = [0]
        
        total_len=0
        #id_sent_len=self.iden_vecs.shape[0] #*self.expression_vecs.shape[0]
        id_sent_len=0
        
        for id_ in tqdm(self.ict_data_synth[self.mode], desc='ict synth'):
            
            curr_v_dir = f"{self.ict_dir}/{id_}/vertices_npy"
            v_listdirs = sorted(glob.glob(f"{curr_v_dir}/*.npy"))
            
            curr_n_dir = f"{self.ict_dir}/{id_}/normals_npy"        
            n_listdirs = sorted(glob.glob(f"{curr_n_dir}/*.npy"))
            
            assert len(v_listdirs) == len(n_listdirs), f"{id_}-{sen_}"
            
            self.ict_synth_v_npy_list[id_]=v_listdirs
            self.ict_synth_n_npy_list[id_]=n_listdirs

            total_len += len(self.ict_synth_v_npy_list[id_])
            
            # remain = len(v_listdirs) % self.opts.batch_size
            remain = 1
            if remain > 0:
                dummy = self.opts.batch_size - remain
            else:
                dummy = 0
            
            n_id_sent = self.opts.batch_size # len(v_listdirs) + dummy
            id_sent_len += n_id_sent # SEN len
            self.ict_synth_count.append(n_id_sent)
            self.ict_synth_remain.append(dummy)
            
        self.ict_synth_count=np.array(self.ict_synth_count)
        self.ict_synth_count=np.cumsum(self.ict_synth_count)
        self.ict_synth_remain=np.array(self.ict_synth_remain)
            
        return total_len, id_sent_len
    
    def set_ict_synth_single(self):
        
        self.iden_vecs = np.load('./ict_face_pt/random_identity_vecs.npy')
        self.expression_vecs = np.load('./ict_face_pt/random_expression_vecs.npy')
                
        self.ict_synth_precompute = f'{self.ict_basedir}/precompute-synth-fullhead'
        self.ict_synth_precompute_fo = f'{self.ict_basedir}/precompute-synth-face_only'
        
        self.ict_dir = os.path.join(self.ict_basedir, 'synth_set', 'train')
        
        self.ict_synth_v_npy_list = {}
        self.ict_synth_n_npy_list = {}
        #self.ict_synth_len_list = {}
        
        total_len=0
        id_ = '100'
        id_sent_len=self.expression_vecs.shape[0]
            
        curr_v_dir=f"{self.ict_dir}/{id_}/vertices_npy"
        curr_n_dir=f"{self.ict_dir}/{id_}/normals_npy"
        
        self.ict_synth_v_npy_list[id_]=sorted(glob.glob(f"{curr_v_dir}/*.npy"))
        self.ict_synth_n_npy_list[id_]=sorted(glob.glob(f"{curr_n_dir}/*.npy"))

        assert len(self.ict_synth_v_npy_list[id_]) == len(self.ict_synth_n_npy_list[id_]), f"{id_}"

        total_len += len(self.ict_synth_v_npy_list[id_])
            #id_sent_len+=1
        return total_len, id_sent_len
    
    def get_ICTcapture(self, index, select):
        # sent_idx = index % self.ict_sent_len
        # id_idx = index // self.ict_sent_len
        
        id_idx = np.where(self.ict_real_count > index)[0][0] - 1
        sent_idx = index - self.ict_real_count[id_idx]
        o_index = self.ict_real_count[id_idx+1] - self.ict_real_count[id_idx] - self.ict_real_remain[id_idx+1]
        if sent_idx >= o_index:
            sent_idx = random.randint(0, o_index -1)
        
        id_key = self.ict_data_split[self.mode][id_idx]
        sent_key = self.ict_sent_key_list[sent_idx]
        
        frame_idx = random.randint(0, len(self.ict_real_v_npy_list[id_key][sent_key]) -1)
        
        # start_time = time.time()
        id_coeff = torch.from_numpy(self.ict_real_id_vecs[id_idx]) # [100]
        id_coeff = torch.cat([id_coeff, torch.zeros(28)]).float() # [128]
        
        exp_coeff = np.load(self.ict_real_exp_coeff_list[id_key][sent_idx])[frame_idx]
        exp_coeff = torch.from_numpy(exp_coeff) # [53]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(75)]).float() # [128]
        # end_time = time.time()
        # print('00time elapsed:', end_time - start_time)
        
        # face only or full
        # region = random.randint(0, 1)
        # v_num, quad_f_num = self.ict_face_model.region[region]
        # f_num = quad_f_num*2
        # if region == 0: # full
        #     precompute_dir = self.ict_real_precompute
        # else: # face only
        #     precompute_dir = self.ict_real_precompute_fo
        v_num, quad_f_num = self.ict_face_model.region[0]
        f_num = quad_f_num*2
        precompute_dir = self.ict_real_precompute
            
        # start_time = time.time()
        ## most bottleneck
        v_normal = np.load(self.ict_real_n_npy_list[id_key][sent_key][frame_idx])[:v_num]
        vertices = np.load(self.ict_real_v_npy_list[id_key][sent_key][frame_idx])[:v_num]
        template = self.ict_real_templates_dict[id_key][:v_num]
        faces = self.ict_real_templates_dict['face'][:f_num]
        # end_time = time.time()
        # print('01time elapsed:', end_time - start_time)
        
        v_normal = torch.from_numpy(v_normal).float()
        vertices = torch.from_numpy(vertices).float()
        template = torch.from_numpy(template).float()
        faces = torch.from_numpy(faces).long()
        
        ## Random Augmentation ---------------------------------------------------
        # template, vertices = self.random_trans_scale(template, vertices)
        ## -----------------------------------------------------------------------
        
        # start_time = time.time()
        dfn_info = os.path.join(precompute_dir, f"{id_key}_dfn_info.pkl")
        operators = os.path.join(precompute_dir, f"{id_key}_operators.pkl")
        # end_time = time.time()
        # print('02time elapsed:', end_time - start_time)
        
        #start_time = time.time()
        img = np.load(os.path.join(precompute_dir, f"{id_key}_img.npy"))
        img = torch.from_numpy(img)[0]
        # end_time = time.time()
        #print('03time elapsed:', end_time - start_time)
        
        # get audio feature + slice w/ window
        dummy = torch.zeros(1)
        
        # v_normal = calc_norm_torch(vertices, faces, at='v').float()
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
        
    def get_ICTsynthetic(self, index, select):
        #e_index = index % self.expression_vecs.shape[0]
        #id_idx = index // self.expression_vecs.shape[0]
        # e_index = random.randint(0, self.expression_vecs.shape[0] -1)

        # id_idx = index
        id_idx = np.where(self.ict_synth_count > index)[0][0] - 1
        e_index = random.randint(0, self.expression_vecs.shape[0] -1)
        # e_index = index - self.ict_synth_count[id_idx]
        # o_index = self.ict_synth_count[id_idx+1] - self.ict_synth_count[id_idx] - self.ict_synth_remain[id_idx+1]
        # if e_index >= o_index:
        #     e_index = random.randint(0, o_index -1)
        
        if self.use_ict_synth_single:
            id_idx = 100 * self.opts.batch_size
            
        # get id_coeff
        id_coeff  = torch.from_numpy(self.iden_vecs[id_idx]) #-------------------- [100]
        id_coeff  = torch.cat([id_coeff, torch.zeros(28)]).float() # ------------- [128]
        
        exp_coeff = torch.from_numpy(self.expression_vecs[e_index]) # ------------ [53]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(75)], dim=-1).float() # ---- [128]
                
        id_key = f'{id_idx:03d}'
        
        #select = random.randint(0, 2)
        v_num, faces = self.ict_face_model.get_random_v_and_f(select=select)
        
        if select == 0:
            precompute_dir = self.ict_synth_precompute
        elif select == 1:
            precompute_dir = self.ict_synth_precompute_fo
        else:
            precompute_dir = self.ict_synth_precompute_nf
        
        # v_num, quad_f_num = self.ict_face_model.region[0]
        # f_num = quad_f_num*2
        # precompute_dir = self.ict_synth_precompute
        
        v_normal = np.load(self.ict_synth_n_npy_list[id_key][e_index])[:v_num]
        vertices = np.load(self.ict_synth_v_npy_list[id_key][e_index])[:v_num]
        template = self.ict_synth_templates_dict[id_key][:v_num]
        #faces = self.ict_synth_templates_dict['face'][:f_num]
        
        v_normal = torch.from_numpy(v_normal).float()
        vertices = torch.from_numpy(vertices).float()
        template = torch.from_numpy(template).float()
        faces = torch.from_numpy(faces).long()
        
        ## Random Augmentation ---------------------------------------------------
        # template, vertices = self.random_trans_scale(template, vertices)
        ## -----------------------------------------------------------------------
        
        dfn_info = os.path.join(precompute_dir, f"{id_key}_dfn_info.pkl")
        operators = os.path.join(precompute_dir, f"{id_key}_operators.pkl")
        
        img = np.load(os.path.join(precompute_dir, f"{id_idx:03d}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        dummy = torch.zeros(1)
        
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
        
    def get_multiface_SEN(self, index):
        
        id_idx = np.where(self.mf_SEN_count > index)[0][0] # num begins from 1
        e_index = index - self.mf_SEN_count[id_idx-1]
        o_index = self.mf_SEN_count[id_idx] - self.mf_SEN_count[id_idx-1] - self.mf_SEN_remain[id_idx]
        if e_index >= o_index:
            e_index = random.randint(0, o_index -1)

        id_name = self.mf_data_split[self.mode][id_idx-1] # num begins from 0
        SEN = list(self.mf_SEN_v_npy_list[id_name].keys())[e_index]

        frame_idx = random.randint(0, len(self.mf_SEN_v_npy_list[id_name][SEN]) -1)
        template = self.mf_verts[id_idx-1] # 5223
        # self.mf_std['v_idx'] # 5223
        vertices = np.load(self.mf_SEN_v_npy_list[id_name][SEN][frame_idx]) # 5223
        v_normal = np.load(self.mf_SEN_n_npy_list[id_name][SEN][frame_idx]) # 5223

        npy_file = self.mf_SEN_v_npy_list[id_name][SEN][frame_idx].replace('.npy', '')
        faces = self.mf_std['new_f']
        
        # get template dfn_info (neutral face)
        dfn_info = os.path.join(self.mf_precompute_path, f"{id_name}_dfn_info.pkl")
        operators = os.path.join(self.mf_precompute_path, f"{id_name}_operators.pkl")
        
        img = np.load(os.path.join(self.mf_precompute_path, f"{id_name}_img.npy"))
        img = torch.from_numpy(img)[0]
                
        vertices = torch.from_numpy(vertices).float()
        v_normal = torch.from_numpy(v_normal).float()
        template = torch.from_numpy(template).float()

        ## Random Augmentation ---------------------------------------------------
        # template, vertices = self.random_trans_scale(template, vertices)
        ## -----------------------------------------------------------------------
        
        # get exp_coeff  (no GT == zeros!)
        exp_coeff= torch.zeros(self.WS, 128).float()
        
        # get id_coeff (no GT == zeros!)
        id_coeff = torch.zeros(128).float()
        
        dummy = torch.zeros(1)
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
    
    def get_multiface_ROM(self, index):
        """
        always use neutral face at 0th 
        """
        # e_index = index % self.expression_vecs.shape[0]
        # id_idx = index // self.expression_vecs.shape[0]
        # for id_idx, s in enumerate(self.mf_ROM_count):
        #     if s > index:
        #         break
        id_idx = np.where(self.mf_ROM_count > index)[0][0] # num begins from 1
        e_index = index - self.mf_ROM_count[id_idx-1]
        o_index = self.mf_ROM_count[id_idx] - self.mf_ROM_count[id_idx-1] - self.mf_ROM_remain[id_idx]
        if e_index >= o_index:
            e_index = random.randint(0, o_index -1)

        id_name = self.mf_data_split[self.mode][id_idx-1] # num begins from 0
        ROM = list(self.mf_ROM_v_npy_list[id_name].keys())[e_index]

        frame_idx = random.randint(0, len(self.mf_ROM_v_npy_list[id_name][ROM]) -1)
        template = self.mf_verts[id_idx-1] # 5223
        # self.mf_std['v_idx'] # 5223
        vertices = np.load(self.mf_ROM_v_npy_list[id_name][ROM][frame_idx]) # 5223
        v_normal = np.load(self.mf_ROM_n_npy_list[id_name][ROM][frame_idx]) # 5223

        npy_file = self.mf_ROM_v_npy_list[id_name][ROM][frame_idx].replace('.npy', '')
        # os.path.join(self.mf_obj_path,id_name+'_mesh.obj')
        faces = self.mf_std['new_f']

        # get template dfn_info (neutral face)
        dfn_info = os.path.join(self.mf_precompute_path, f"{id_name}_dfn_info.pkl")
        operators = os.path.join(self.mf_precompute_path, f"{id_name}_operators.pkl")
        # dfn_info  = pickle.load(open(os.path.join(
        #     self.mf_precompute_path, 
        #     f"{id_name}_dfn_info.pkl"
        # ), 'rb'))
        # operators = os.path.join(
        #     self.mf_precompute_path,
        #     f"{id_name}_operators.pkl"
        # )
        img = np.load(os.path.join(self.mf_precompute_path, f"{id_name}_img.npy"))
        img = torch.from_numpy(img)[0]
        
        vertices = torch.from_numpy(vertices).float()
        v_normal = torch.from_numpy(v_normal).float()
        template = torch.from_numpy(template).float()
        
        ## Random Augmentation ---------------------------------------------------
        # template, vertices = self.random_trans_scale(template, vertices)
        ## -----------------------------------------------------------------------
        
        # get id_coeff and exp_coeff (no GT == zeros!)
        id_coeff = torch.zeros(128)
        exp_coeff = torch.zeros(self.WS, 128)
        dummy = torch.zeros(1)
        
        # v_normal = calc_norm_torch(vertices, faces, at='v')
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
    
    def get_voca(self, index):
        
        id_idx = np.where(self.voca_count > index)[0][0] # num begins from 1
        e_index = index - self.voca_count[id_idx-1]
        o_index = self.voca_count[id_idx] - self.voca_count[id_idx-1] - self.voca_remain[id_idx]
        if e_index >= o_index:
            e_index = random.randint(0, o_index -1)
            
        id_name = self.voca_data_split[self.mode][id_idx-1] # num begins from 0
        sent = list(self.voca_v_npy_list[id_name].keys())[e_index]
        
        frame_idx = random.randint(0, len(self.voca_v_npy_list[id_name][sent]) -1)
        
        # get template vertices (neutral face)
        template = self.voca_coma_templates[id_name] # 3525, 3
        vertices = np.load(self.voca_v_npy_list[id_name][sent][frame_idx]) # 3525, 3
        v_normal = np.load(self.voca_n_npy_list[id_name][sent][frame_idx]) # 3525, 3
        
        faces = self.voca_coma_std['new_f'] ## removed eyeball
        
        # get template dfn_info (neutral face)
        dfn_info  = os.path.join(self.voca_coma_precompute_path, f"{id_name}_dfn_info.pkl")
        operators = os.path.join(self.voca_coma_precompute_path, f"{id_name}_operators.pkl")
        
        img = np.load(os.path.join(self.voca_coma_precompute_path, f"{id_name}_img.npy")) # ----------------- [1, 256, 256, 3]
        img = torch.from_numpy(img)[0].float()
                    
        vertices = torch.from_numpy(vertices).float()
        v_normal = torch.from_numpy(v_normal).float()
        template = torch.from_numpy(template).float()
        faces = torch.from_numpy(faces).long()

        id_coeff = torch.zeros(128)
        exp_coeff = torch.zeros(self.WS, 128)
        dummy = torch.zeros(1)
        
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
        
    def get_coma(self, index):
        
        id_idx = np.where(self.coma_count > index)[0][0] # num begins from 1
        e_index = index - self.coma_count[id_idx-1]
        o_index = self.coma_count[id_idx] - self.coma_count[id_idx-1] - self.coma_remain[id_idx]
        if e_index >= o_index:
            e_index = random.randint(0, o_index -1)
            
        id_name = self.voca_data_split[self.mode][id_idx-1] # num begins from 0
        ROM = list(self.coma_v_npy_list[id_name].keys())[e_index]
        
        frame_idx = random.randint(0, len(self.coma_v_npy_list[id_name][ROM]) -1)
        
        # get template vertices (neutral face)
        template = self.voca_coma_templates[id_name] # 3525, 3
        vertices = np.load(self.coma_v_npy_list[id_name][ROM][frame_idx]) # 3525, 3
        v_normal = np.load(self.coma_n_npy_list[id_name][ROM][frame_idx]) # 3525, 3
        
        faces = self.voca_coma_std['new_f'] ## removed eyeball
        
        # get template dfn_info (neutral face)
        dfn_info  = os.path.join(self.voca_coma_precompute_path, f"{id_name}_dfn_info.pkl")
        operators = os.path.join(self.voca_coma_precompute_path, f"{id_name}_operators.pkl")
        
        img = np.load(os.path.join(self.voca_coma_precompute_path, f"{id_name}_img.npy")) # ----------------- [1, 256, 256, 3]
        img = torch.from_numpy(img)[0].float()

        vertices = torch.from_numpy(vertices).float()
        v_normal = torch.from_numpy(v_normal).float()
        template = torch.from_numpy(template).float()
        faces = torch.from_numpy(faces).long()

        id_coeff = torch.zeros(128)
        exp_coeff = torch.zeros(self.WS, 128)
        dummy = torch.zeros(1)
        
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img

    def get_biwi(self, index):
        
        id_idx = np.where(self.biwi_count > index)[0][0] # num begins from 1
        e_index = index - self.biwi_count[id_idx-1]
        o_index = self.biwi_count[id_idx] - self.biwi_count[id_idx-1] - self.biwi_remain[id_idx]
        if e_index >= o_index:
            e_index = random.randint(0, o_index -1)
        
        id_name = self.biwi_id[id_idx-1] # num begins from 0
        sent = list(self.biwi_v_npy_list[id_name].keys())[e_index]
        
        frame_idx = random.randint(0, len(self.biwi_v_npy_list[id_name]) -1)
        
        # get template vertices (neutral face)
        template = self.biwi_templates[id_name] # 2560, 3
        vertices = np.load(self.biwi_v_npy_list[id_name][sent][frame_idx]) # 2560, 3
        v_normal = np.load(self.biwi_n_npy_list[id_name][sent][frame_idx]) # 2560, 3
        
        # get template face indices
        faces = self.biwi_templates['face']
        
        # get template dfn_info (neutral face)
        dfn_info = os.path.join(self.biwi_precompute_path, f"{id_name}_dfn_info.pkl")
        operators = os.path.join(self.biwi_precompute_path, f"{id_name}_operators.pkl")
        
        img = np.load(os.path.join(self.biwi_precompute_path, f"{id_name}_img.npy")) # ----------------- [1, 256, 256, 3]
        img = torch.from_numpy(img)[0]
        
        vertices = torch.from_numpy(vertices).float()
        v_normal = torch.from_numpy(v_normal).float()
        template = torch.from_numpy(template).float()
        faces = torch.from_numpy(faces).long()
        
        id_coeff = torch.zeros(128)
        exp_coeff = torch.zeros(self.WS, 128)
        dummy = torch.zeros(1)
        
        return dummy, id_coeff, exp_coeff, template, dfn_info, operators, vertices, v_normal, faces, img
            
    def random_rotation_matrix(self, randgen=None):
        """
        Borrowed from https://github.com/nmwsharp/diffusion-net/blob/master/src/diffusion_net/utils.py
        
        Creates a random rotation matrix.
        randgen: if given, a np.random.RandomState instance used for random numbers (for reproducibility)
        """
        # adapted from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
        
        if randgen is None:
            randgen = np.random.RandomState()
            
        theta, phi, z = tuple(randgen.rand(3).tolist())
        
        theta = theta * 2.0*np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0*np.pi  # For direction of pole deflection.
        z = z * 2.0 # For magnitude of pole deflection.
        
        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.
        
        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
            )
        
        st = np.sin(theta)
        ct = np.cos(theta)
        
        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M
    
    def random_rotate_points(self, pts, randgen=None):
        R = self.random_rotation_matrix(randgen) 
        R = torch.from_numpy(R).to(device=pts.device, dtype=pts.dtype)
        return torch.matmul(pts, R)
    
    def random_trans_scale(self, template, vertices):
        ## Random Augmentation ---------------------------------------------------------
        trans, scale = 0.0, 1.0
        if self.opts.data_rand_trans:
            t_range = 0.25
            trans = (torch.rand((1, 3))*t_range - t_range*0.5)
        # if self.opts.data_rand_scale:
        #     scale = torch.rand((1)).repeat(3) * 0.4 + 0.8
        template = template * scale + trans
        vertices = vertices * scale + trans
        ## -----------------------------------------------------------------------------
        
        return template, vertices
    
    def __getitem__(self, index):
        select = index[0]
        idx = index[1]
        
        #select = random.randint(0, 2)
        for _, get_data, mesh_data, id_sent_len in self.len_list:
            if idx < id_sent_len:
                datas = get_data(idx, select) if mesh_data == 0 else get_data(idx)
                break
            else:
                idx = idx - id_sent_len
        return (*datas, mesh_data)
    
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
        elif self.use_voca and template.shape[0] == self.voca_coma_std['v_idx'].shape[0]:
            return self.identity_num[audio_path.split(self.mode)[-1].split('/')[1]]
        
        elif self.use_biwi and template.shape[0] == self.biwi_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('self.mode')[-1].split('/')[-1].split('_')[0]]
        
        # elif self.use_mf_ROM and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
        #     return
        
        elif self.use_mf_SEN and template.shape[0] == self.mf_trimesh.vertices.shape[0]:
            return self.identity_num[audio_path.split('wav2vec2')[-1].split('/')[-1].split('-SEN')[0]]
    
    def vis_mesh(self, 
                 vertices, # [B, V, 3]
                 faces=None, 
                #  frame=0, 
                 mesh='ict', 
                 tag='', 
                 bg_black=False,
                 size=3,
                 render_mode='shade',
                 logdir='_tmp',
                ):
        if faces is None:
            if mesh == 'ict':
                if vertices.shape[1] == 11248:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=0)
                elif vertices.shape[1] == 9409:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=1)
                else:
                    _, faces = self.ict_face_model.get_random_v_and_f(select=2)
            elif mesh == 'voca':
                #faces = self.voca_trimesh.faces
                faces = self.voca_coma_std['new_f']
            elif mesh == 'biwi':
                faces = self.biwi_templates['face']
            elif mesh == 'mf':
                faces = self.mf_std['new_f']
            
        v_list = vertices
        len_v = len(v_list)
        
        f_list = [faces]*len_v
        rot_list=[[0,0,0]]*len_v
        
        os.makedirs(logdir, exist_ok=True)
        
        plot_image_array(
            v_list, f_list, rot_list, 
            size=size,
            mode=render_mode,
            bg_black=bg_black, 
            logdir=logdir,
            save=True,
            name=f'{tag}-{mesh}'
        )

# deprecated
class NFSDataset(data.Dataset):
    def __init__(self, 
                 opts,
                 ict_basedir='/data/sihun/ICT-audio2face/split_set/', 
                 mf_basedir="/data/sihun/multiface/audio2face",
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
        self.ict_data_split, self.voca_data_split, self.biwi_data_split, self.mf_data_split, _ = get_data_splits()
        self.identity_num = get_identity_num()
        
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
                 ict_basedir='/data/ICT-audio2face/split_set/', 
                 voca_basedir="/data/VOCASET/original_set",
                 biwi_basedir="/data/BIWI",
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
        super(InvRigDataset, self).__init__()
        # get basenames
        self.opts = opts
        self.is_train = is_train
        self.is_valid = is_valid
        self.WS = self.opts.window_size if self.opts is not None else window_size
        self.device = device
        self.audio_feat_level = audio_feat_level
        self.audio_feat_type = audio_feat_type
        
        self.ict_face_only = self.opts.ict_face_only if self.opts is not None else ict_face_only
        
        self.mode = 'test'
        if is_train:
            self.mode = 'train'
        elif is_valid:
            self.mode = 'val'
            
        ## precomputes --------------------------------------------------------------------
        p_mode = 'face_only' if self.ict_face_only else 'fullhead'
            
        self.ict_data_split = {
            'train': ['m00', 'm02', 'm04', 'm05', 'm06', 
                      'w00', 'w01', 'w02', 'w05', 'w08'], # 00 ~ 60
            'val':   ['m00', 'm02', 'm04', 'm05', 'm06',
                      'w00', 'w01', 'w02', 'w05', 'w08'], # 60 ~ 65
            'test':  ['m00', 'm02', 'm04', 'm05', 'm06',
                      'w00', 'w01', 'w02', 'w05', 'w08'], # 66 ~ 72
        }
                
        self.only_ict = False
        ## ICT-facekit --------------------------------------------------------------------
        self.ict_face_model = ICT_face_model(face_only=self.ict_face_only)
        self.ict_templates_path = './ICT/templates'
        
        self.iden_vecs = np.load('./data/ICT_live_100/iden_vecs.npy')[:8]
        self.expression_vecs = np.load(f'./data/ICT_live_100/expression_vecs_{self.mode}.npy')[:128]
        self.ict_precompute_path = f'./ICT/precompute-{p_mode}'
        self.ict_precompute_path_synth = f'./ICT/precompute-synth-{p_mode}'
        
        self.ict_basedir = os.path.join(ict_basedir,  self.mode)
        self.ict_audio_wav = sorted(glob.glob(f'{self.ict_basedir}/*/wav/*.wav'))
        
        self.ict_vert_segment = torch.from_numpy(np.load('./mesh_utils/ict/ICT_segment_onehot.npy'))
        #zzz = zzz.argmax(1)
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
        id_idx = index // self.expression_vecs.shape[0]
        exp_idx = index % self.expression_vecs.shape[0]
        
        # get id_coeff
        id_coeff  = torch.from_numpy(self.iden_vecs[id_idx]).float() # [100]
        id_coeff  = torch.cat([id_coeff, torch.zeros(28)]) # --------------------------------------------- [128]
        
        exp_coeff = torch.from_numpy(self.expression_vecs).float()[None, exp_idx] # -------- [1, 53]
        exp_coeff = torch.cat([exp_coeff, torch.zeros(1, 75)], dim=-1) # ---------------------------- [W, 128]
        
        id_disps  = torch.einsum('k,kls->ls', id_coeff[:100], self.ict_face_model.id_basis)[:self.ict_face_model.v_idx]
        exp_disp  = torch.einsum('jk,kls->jls', exp_coeff[:, :53], self.ict_face_model.exp_basis)[:,:self.ict_face_model.v_idx]
        
        # get template vertices (neutral face)
        template  = self.ict_face_model.neutral_verts + id_disps
        faces     = self.ict_face_model.faces
        
        # get animation vertices using exp_coeff
        vertices  = template + exp_disp
        
        # get template dfn_info (neutral face)
        dfn_info  = pickle.load(open(os.path.join(self.ict_precompute_path_synth, f"{id_idx:03d}_dfn_info.pkl"), 'rb')) # list[ ... ]
        operators = os.path.join(self.ict_precompute_path_synth, f"{id_idx:03d}_operators.pkl")
        
        img       = np.load(os.path.join(self.ict_precompute_path_synth, f"{id_idx:03d}_img.npy")) # ----- [1, 256, 256, 3]
        img       = torch.from_numpy(img)[0]
        
        audio_feat= torch.zeros(self.WS, 768)
        
        return audio_feat, id_coeff, exp_coeff, template, dfn_info, operators, vertices, faces, img
        
        
    def __getitem__(self, index):
        return self.get_ICTsynthetic(index), torch.tensor(0)
        
class MeshSampler(data.Sampler):
    def __init__(self, len_list, batch_size, shuffle=False, balance=False, n_sampling=False, n_=4,reverse=False):
        self.len_list = len_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.reverse = reverse
        self.balance = balance
        self.mode = np.array(['ict', 'voca', 'biwi', 'mf'])
        self.n_sampling = n_sampling
        self.n_ = n_
        self.epoch = 0
        self.indices, self.labels = self._get_indices()
        self.total = self.indices.shape[0]
        
    def _get_indices(self):
        """
        Returns:
            indices (np.ndarray): indices array for the data [N, Batch]
            labels (np.ndarray): labels for the data, just for debugging
        """
        indices = np.zeros(0, dtype=int)
        labels = np.zeros(0, dtype=int) # for debugging
        SS = 0
        for len_data, _, mesh_data, id_len_list in self.len_list:
            
            m_data = mesh_data.numpy()
            _indices = np.arange(SS, SS+id_len_list, dtype=int)
            id_len_list_tmp = id_len_list
            
            if m_data == 1: # voca or coma
                tile_n = 10
                _indices = np.tile(_indices, tile_n)
                id_len_list_tmp = id_len_list_tmp * tile_n
            if m_data == 2: # biwi
                tile_n = 7
                _indices = np.tile(_indices, tile_n)
                id_len_list_tmp = id_len_list_tmp * tile_n
            if m_data == 3: # mf
                tile_n = 2
                _indices = np.tile(_indices, tile_n)
                id_len_list_tmp = id_len_list_tmp * tile_n
            
            if self.n_sampling:
                _remain = id_len_list_tmp % (self.batch_size * self.n_)
            else:
                _remain = id_len_list_tmp % self.batch_size
                
            if _remain > 0:
                indices = np.r_[indices, _indices[0:-_remain]]
                labels = np.r_[labels, np.ones(id_len_list_tmp-_remain) * m_data]
            else:
                indices = np.r_[indices, _indices]
                labels = np.r_[labels, np.ones(id_len_list_tmp) * m_data]
            SS = SS + id_len_list
        
        if self.n_sampling:
            indices = indices.reshape(-1, self.batch_size, self.n_).transpose(0, 2, 1).reshape(-1, self.batch_size)
            labels = labels.reshape(-1, self.batch_size, self.n_).transpose(0, 2, 1).reshape(-1, self.batch_size)
        else:
            indices = indices.reshape(-1, self.batch_size)
            labels = labels.reshape(-1, self.batch_size)
        
        assert indices.shape[0] == labels.shape[0], "miss match!"
                
        return indices, labels
    
    def __iter__(self):
        # TODO:
        # 1. (inner) 각 데이터 길이를 batch size의 배수로 만들기 (일부 중복 추가)
        # 2. (inner) 각 데이터를 셔플
        # 3. (inner) batch_size 에 맞게 각 데이터 재 배열
        # 4. (cross) 데이터 별로 섞기
        
        # if self.n_sampling:
        #     idx = np.arange(self.epoch % 3, self.total, 3)
        # else:
        #     idx = np.arange(self.indices.shape[0])
        idx = np.arange(self.indices.shape[0])
            
        if self.shuffle:
            idx = np.random.permutation(idx)
            # self.indices = self.indices[idx]
            # self.labels = self.labels[idx]
            indices = self.indices[idx]
            labels = self.labels[idx]
        else:
            indices = self.indices
            labels = self.labels
            
        if self.reverse:
            indices = self.indices[::-1]
            labels = self.labels[::-1]
        
        #batch = indices.tolist()
        # self.length = len(batch)
        select = np.tile(np.random.randint(3, size=indices.shape[0]), 8).reshape(8,-1).transpose()
        batch = np.concatenate([select[:,:,None], indices[:,:,None]], axis=-1)
        batch = batch.tolist()
        
        self.length = len(indices)
        
        return iter(batch)

    def __len__(self):
        # if self.n_sampling:
        #     base_len = self.total // 3
        #     remainder = self.total % 3     
        #     if self.epoch % 3 == 0:
        #         return base_len + (1 if remainder > 0 else 0)
        #     elif self.epoch % 3 == 1:
        #         return base_len + (1 if remainder > 1 else 0) 
        #     else:
        #         return base_len
        # else:
        #     return len(self.indices)
        return self.total
    
    def get_sampler_config(self):
        text = "=========[MeshSampler]=========\n"
        text += f"[Batch size]: {self.batch_size}\n"
        for i, mode in enumerate(self.mode):
            mode_len = len(np.where(self.labels[:,0]==i)[0])
            if self.balance and (mode == 'biwi' or mode == 'voca'):
                text += f"[Batched {mode.upper()}]: {mode_len} (multiplied)\n"
            else:
                text += f"[Batched {mode.upper()}]: {mode_len}\n"
        text += f"[Batched len]: {len(self.indices)}\n"
        text += "===============================\n"
        return text
    
    def set_epoch(self, epoch):
        self.epoch = epoch

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
            self.template = torch.stack(transposed_data[3], 0)
            self.normals = torch.stack(transposed_data[7], 0)
            
            self.img = torch.stack(transposed_data[9], 0)
            
            if len(transposed_data) > 11:
                self.audio_path = transposed_data[11]
            
            # pickle path, same mesh in minibatch == same operator!
            self.mesh_data = transposed_data[10][0]
            #self.template = transposed_data[3][0][None]
            
            self.faces = transposed_data[8][0]
            self.dfn_info = transposed_data[4][0]
            self.operators = transposed_data[5][0]
            #self.set_dfn_info(transposed_data[4][0])
            
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
    
    opts.window_size = 1
    print(f'use window_size: {opts.window_size}')
    
    opts.batch_size = 8
    print(f'use batch_size: {opts.batch_size}')
    
    #dataset = InvRigDataset(opts, is_train=True)
    #dataset = MeshDataset(opts, is_train=True, is_valid=False)
    dataset = MeshDataset(opts, is_train=False, is_valid=True)
    # dataset = MeshDataset(opts, is_train=False, is_valid=False)
    
    # total_len, id_sent_len = dataset.set_multiface_SEN()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_multiface_ROM()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_voca()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_coma()
    # print(total_len, id_sent_len)
    
    # total_len, id_sent_len = dataset.set_biwi()
    # print(total_len, id_sent_len)
    
    # dataset.biwi_len_list
    # total_len, id_sent_len = dataset.set_ict_synth()
    # print(total_len, id_sent_len)
    # total_len, id_sent_len = dataset.set_ict_real()
    # print(total_len, id_sent_len)
    print(dataset.get_data_config())
    # dataset.ict_real_len_list
    
    # dataset = NFSDataset(opts, is_train=True, return_audio_dir=True)
    # print(dataset.get_data_config())
    
    sampler = MeshSampler(
        dataset.len_list, 
        opts.batch_size,
        #shuffle=True,
        balance=False,
        n_sampling=opts.n_sampling,
        n_=8,
        reverse=True,
    )
    print('n_sampling', opts.n_sampling)
    print(sampler.get_sampler_config())
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_sampler=sampler, 
        collate_fn=partial(collate_wrapper, device=opts.device),
        num_workers=0
    )
    
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, 
    #     batch_size=opts.batch_size, 
    #     shuffle=False,
    #     collate_fn=partial(collate_wrapper, device=opts.device), 
    #     num_workers=0
    # )
    len_dataloader = len(dataloader)
    #data = next(iter(dataloader))
    # iter_dataloader = iter(dataloader)
    # pbar = tqdm(enumerate(dataloader), total=len_dataloader, position=0)
    # rambar = tqdm(range(len_dataloader), total=125, desc='ram', position=1, ncols=100)
    # cpubar = tqdm(range(len_dataloader), total=100, desc='cpu', position=2, ncols=100)
    
    
    from utils import plot_image_array, plot_image_array_diff3, vis_rig 
    
    pbar = tqdm(enumerate(dataloader), total=len_dataloader)
    for idx, batch in pbar:
        #(audio_feat, id_coeff, gt_rig_params, template, dfn_info, operators, vertices, faces, img), mesh_data, audio_path = batch
        #print(audio_path[0], audio_feat.shape, id_coeff.shape, gt_rig_params.shape, template.shape, vertices.shape)
        
        mode = np.array(['ict', 'voca', 'biwi', 'mf'])[batch.mesh_data]
        # print(idx, mode, batch.audio_feat.shape, batch.gt_rig_params.shape, batch.template.shape, batch.vertices.shape, batch.normals.shape)
        pbar.set_description(f"{idx}-{mode}")
        # plot_image_array(
        #         v_list, f_list, 
        #         rot_list=[[0,0,0]] * len_v, 
        #         size=1, bg_black=False, mode='shade',
        #         logdir=save_logdir,
        #         name=save_img_name, save=True
        #     )
        #print(template.min(), template.max())
        dataset.vis_mesh(batch.vertices.cpu(), mesh=mode,tag=f"{idx:06d}")
        # dataset.vis_mesh(batch.template.cpu(), frame=1, mesh='ict')
        # dataset.vis_mesh(batch.vertices.cpu(), frame=0, mesh='ict')
        # dataset.vis_mesh(batch.vertices.cpu(), frame=1, mesh='ict')
        # dataset.vis_mesh(batch.vertices.cpu(), frame=3, mesh='ict')
