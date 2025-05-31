# Original code from: 
# https://github.com/dafei-qin/NFR_pytorch/blob/master/mymodel.py

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pickle

import sys
import time
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path, f"{abs_path}/third_party/diffusion-net/src"]
import diffusion_net

from .CNN import TextureEncoder
from .decoder import *
from .encoder import *

from utils import (
    ICT_face_model, 
    calc_norm_torch,
    nfr_utils
)
from utils.deformation_transfer import deformation_gradient

import logging


class my_diffusion_net_template(nn.Module):
    def __init__(self, in_shape, out_shape, hid_shape, pre_computes, N_block=4, outputs_at='faces', with_grad=True):
        super(my_diffusion_net_template, self).__init__()
        self.dfn = diffusion_net.DiffusionNet(C_in=in_shape, C_out=out_shape, C_width=hid_shape, N_block=N_block, outputs_at=outputs_at, with_gradient_features=with_grad)
        self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

        self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
        self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
        self.L_size = pre_computes[1].size()
        self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
        self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
        self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
        self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
        self.grad_X_size =pre_computes[4].size()
        self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
        self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
        self.grad_Y_size = pre_computes[5].size()

        self.faces = nn.Parameter(pre_computes[6], requires_grad=False)

    def update_precomputes(self, pre_computes):
        if len(pre_computes[0].shape) > 1:
            self.mass = nn.Parameter(pre_computes[0].squeeze(0), requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices()[1:], requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()[1:]
            self.evals = nn.Parameter(pre_computes[2].squeeze(0), requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3].squeeze(0), requires_grad=False)
            self.grad_X_ind  = nn.Parameter(pre_computes[4]._indices()[1:], requires_grad=False)
            self.grad_X_val  = nn.Parameter(pre_computes[4]._values(),  requires_grad=False)
            self.grad_X_size = pre_computes[4].size()[1:]
            self.grad_Y_ind  = nn.Parameter(pre_computes[5]._indices()[1:], requires_grad=False)
            self.grad_Y_val  = nn.Parameter(pre_computes[5]._values(),  requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()[1:]

            self.faces = nn.Parameter(pre_computes[6].long(), requires_grad=False)
            
        else:
            self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

            self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
            self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
            self.L_size = pre_computes[1].size()
            self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
            self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
            self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
            self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
            self.grad_X_size =pre_computes[4].size()
            self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
            self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
            self.grad_Y_size = pre_computes[5].size()

            self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)
        # self.mass = nn.Parameter(pre_computes[0], requires_grad=False)

        # self.L_ind = nn.Parameter(pre_computes[1]._indices(), requires_grad=False)
        # self.L_val = nn.Parameter(pre_computes[1]._values(), requires_grad=False)
        # self.L_size = pre_computes[1].size()
        # self.evals = nn.Parameter(pre_computes[2], requires_grad=False)
        # self.evecs = nn.Parameter(pre_computes[3], requires_grad=False)
        # self.grad_X_ind = nn.Parameter(pre_computes[4]._indices(), requires_grad=False)
        # self.grad_X_val = nn.Parameter(pre_computes[4]._values(), requires_grad=False)
        # self.grad_X_size =pre_computes[4].size()
        # self.grad_Y_ind = nn.Parameter(pre_computes[5]._indices(), requires_grad=False)
        # self.grad_Y_val = nn.Parameter(pre_computes[5]._values(), requires_grad=False)
        # self.grad_Y_size = pre_computes[5].size()

        # self.faces = nn.Parameter(pre_computes[6].unsqueeze(0).long(), requires_grad=False)

    def forward(self,
                inputs,
                batch_mass=None,
                batch_L_val=None,
                batch_evals=None,
                batch_evecs=None,
                batch_gradX=None,
                batch_gradY=None):

        self.L = torch.sparse_coo_tensor(self.L_ind, self.L_val, self.L_size, device=inputs.device)
        batch_size = inputs.shape[0]
        if batch_mass is not None:
            batch_L = [torch.sparse_coo_tensor(self.L_ind, batch_L_val[i], self.L_size, device=inputs.device) for i in range(len(batch_L_val))]
        else:
            
            batch_L = [self.L for b in range(batch_size)]
            batch_mass = self.mass.unsqueeze(0).expand(batch_size, -1)
            batch_evals = self.evals.unsqueeze(0).expand(batch_size, -1)
            batch_evecs = self.evecs.unsqueeze(0).expand(batch_size, -1, -1)

        if batch_gradX is not None:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, gX, self.grad_X_size, device=inputs.device) for gX in batch_gradX ]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, gY, self.grad_Y_size, device=inputs.device) for gY in batch_gradY ]
        else:
            gradX = [torch.sparse_coo_tensor(self.grad_X_ind, self.grad_X_val, self.grad_X_size, device=inputs.device) for b in range(batch_size)]
            gradY = [torch.sparse_coo_tensor(self.grad_Y_ind, self.grad_Y_val, self.grad_Y_size, device=inputs.device) for b in range(batch_size)]
        outputs = self.dfn(inputs, batch_mass, L=batch_L, evals=batch_evals, evecs=batch_evecs, gradX=gradX, gradY=gradY, faces=self.faces)
        return outputs

class mymodel(nn.Module):
    def __init__(self, in_shape, hid_shape=128, use_lipschitz=False):
        super(mymodel, self).__init__()

        linear_layer = nn.Linear
        self.model = nn.Sequential(
                linear_layer(in_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, hid_shape),
                nn.ReLU(),
                linear_layer(hid_shape, 9)
            )

    def __call__(self, input):
        return self.model(input)

class latent_space(nn.Module):
    def __init__(self, 
                gn_in, 
                in_shape,
                out_shape,
                pre_computes=None,
                hid_shape=128,
                latent_shape=32,
                global_pn_shape=None,
                dfn_blocks=4,
                iden_blocks=4,
                non_linear='relu',
                residual=False,
                global_pn=False,
                sampling=0,
                number_gn=32,
                img_encoder=False,
                img_feat=32,
                img_only_mlp=False,
                img_warp=False,
                global_encoder_grad=True,
                iden_encoder_grad=True):
        super(latent_space, self).__init__()
        linear_layer = nn.Linear
        if non_linear == 'relu':
            self.relu = nn.ReLU()
        elif non_linear == 'leaky_relu':
            self.relu = nn.LeakyReLU()
        else:
            logging.error(f'{non_linear} not implemented! Choices from [relu, leaky_relu]')

        self.residual = residual
        self.sampling = sampling
        self.img_enc_type = img_encoder
        self.img_only_mlp = img_only_mlp
        self.img_warp = img_warp
        self.latent_shape = latent_shape
        # Img encoder
        if self.img_enc_type == 'cnn':
            self.img_encoder = TextureEncoder()
            self.img_fc = linear_layer(128, img_feat)
            if not self.img_only_mlp and not self.img_warp:
                gn_in += img_feat
            if not self.img_warp:
                in_shape += img_feat
        else:
            self.img_encoder = None

        # Expression encoder
        self.encoder = my_diffusion_net_template(in_shape=gn_in,
                                                out_shape=latent_shape,
                                                hid_shape=hid_shape,
                                                pre_computes=pre_computes,
                                                N_block=dfn_blocks,
                                                outputs_at='global_mean',
                                                with_grad=global_encoder_grad)

        # Identity encoder
        if global_pn:
            if global_pn_shape == None:
                global_pn_shape = latent_shape

            self.global_pn = my_diffusion_net_template(in_shape=gn_in, 
                                                       out_shape=global_pn_shape,
                                                       hid_shape=hid_shape,
                                                       pre_computes=pre_computes, 
                                                       N_block=iden_blocks, 
                                                       outputs_at='global_mean', 
                                                       with_grad=iden_encoder_grad)

        else:
            self.global_pn = None

        # Warper:
        if self.img_warp:
            self.warper = linear_layer(img_feat + global_pn_shape, 3)

        # Useless...
        self.fc_mu = linear_layer(latent_shape, latent_shape)
        self.fc_var = linear_layer(latent_shape, latent_shape)

        # MLP

        if global_pn and not self.img_warp:
            self.linears = [linear_layer(in_shape+global_pn_shape + latent_shape, hid_shape, bias=False)]
        else:
            self.linears = [linear_layer(in_shape+ latent_shape, hid_shape, bias=False)]
        for _ in range(6):
            self.linears.append(linear_layer(hid_shape, hid_shape, bias=False))
            

        self.gns = [nn.GroupNorm(number_gn, hid_shape) for _ in range(len(self.linears))]
        self.gns = nn.ModuleList(self.gns)

        self.linears = nn.ModuleList(self.linears)
        self.linear_out = linear_layer(hid_shape, out_shape)

    def update_precomputes(self, dfn_info):

        self.encoder.update_precomputes(dfn_info)

        self.global_pn.update_precomputes(dfn_info)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        See https://github.com/pytorch/examples/blob/main/vae/main.py
        """

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).to("cuda")  
        return eps * std + mu

    def img_feat(self, img):
        if self.img_enc_type == 'cnn':
            return self.img_fc(self.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
        elif self.img_enc_type == 'unet':
            feat_per_pixel, feat_enc = self.img_encoder(img.permute(0, 3, 1, 2))
            return feat_per_pixel

    def forward(self, x_v, x_f, \
                batch_mass=None, batch_L_val=None, batch_evals=None, batch_evecs=None, batch_gradX=None, batch_gradY=None, \
                source_img=None, target_img=None, \
                return_latent=False, latent=None, return_time=False, x_source_v=None, \
                pix2face_idxs=None, pix2face_vals=None, neutral_pix2face_idxs=None, neutral_pix2face_vals=None, face2vertex_idxs=None, face2vertex_vals=None):

        # setting for dims
        N_F = x_f.shape[1] # Number of faces
        N_V = x_v.shape[1] # Number of vertices
        B_S = x_v.shape[0] # Batch size

        # x_f = x_f.expand(B_S, -1, -1)
        if latent is not None:
            latent = latent.to(x_f.device)
        if self.img_encoder is not None:
            if not self.img_only_mlp and not self.img_warp:
                img = torch.cat((source_img, target_img), axis=0)
            else:
                img = source_img
            img = img.permute(0, 3, 1, 2) # Use permute to retain the correct order

            img_feat = self.img_encoder(img[:, :3]) # DO NOT CONSIDER DEPTH NOW
            img_feat = self.img_fc(img_feat)
            if not self.img_warp:
                x_f = torch.cat((x_f, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_F, -1)), dim=-1) # The first index is the source feature
            if not self.img_only_mlp and not self.img_warp:
                x_v = torch.cat((x_v, img_feat[1:].unsqueeze(1).expand(-1, N_V, -1)), dim=-1) # The followings are the target feature
                if x_source_v is not None:
                    x_source_v = torch.cat((x_source_v, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_V, -1)), dim=-1)
            img_feat_return = img_feat
            
        else:
            img_feat_return = None


        # Expression Encoder:
        t = time.time()
        pn_feat_return = self.encoder(x_v, 
                                    batch_mass=batch_mass,  
                                    batch_L_val=batch_L_val, 
                                    batch_evals=batch_evals,
                                    batch_evecs=batch_evecs,
                                    batch_gradX=batch_gradX,
                                    batch_gradY=batch_gradY)        
        if latent is not None:
            pn_feat = latent
        else:
            pn_feat = pn_feat_return
        t_exp = time.time() - t

        # Identity Encoder
        t = time.time()
        if self.global_pn is not None:
            source_feat = self.global_pn(x_source_v, 
                                        None,
                                        None)
            source_feat = source_feat.expand(B_S, -1)
            pn_feat = torch.cat([source_feat, pn_feat], dim=-1)
            pn_feat_return = torch.cat([source_feat, pn_feat_return], dim=-1)
        t_iden = time.time() - t

        # Warper
        if self.img_warp:
            warper_in = torch.cat([source_feat, img_feat[0].unsqueeze(0).unsqueeze(0).expand(-1, N_F, -1)], dim=-1)
            warp_vec = self.warper(warper_in)
            # x_f[..., :3] = x_f[..., :3] + warp_vec
            x_f = torch.cat((x_f[..., :3] + warp_vec, x_f[..., 3:]), axis=-1)
        
        # MLP
        t = time.time()
        z = pn_feat # This is for direct latent-code
        feat = z.unsqueeze(1)
        feat = feat.expand(-1, N_F, -1) 
        x_f = x_f.expand(B_S, -1, -1)
        x_f = torch.cat([x_f, feat], dim=-1)
        out = x_f

        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)
        # print(f'Calculating: {time.time() - t:.6f}s')
        t_linear = time.time() - t

        out = self.linear_out(out)
        self.cached_feat = feat
        if return_time:
            return out, None, None, pn_feat_return, img_feat_return, np.array([t_exp, t_iden, t_linear])
        else:
            return out, None, None, pn_feat_return, img_feat_return

    def encode(self,
            x_v, 
            target_img=None,
            p2f_t_idxs=None,
            p2f_t_vals=None,
            f2v_t_idxs=None,
            f2v_t_vals=None,
            N_F=None, 
            batch_mass=None,
            batch_L_val=None,
            batch_evals=None,
            batch_evecs=None,
            batch_gradX=None,
            batch_gradY=None):
        """
        args:
            x_v: [B x N_V x 6] -  vertex feature
            
        output:
            pn_feat_return: [B x latent_shape]
        """
        #N_V = x_v.shape[1]
        N_B, N_V = x_v.shape[:2]
        # image encoder: CNN
        if not self.img_only_mlp and not self.img_warp:
            if self.img_encoder is not None:
                img = target_img.permute(0, 3, 1, 2)
                img_feat = self.img_encoder(img[:, :3, ...]) # DO NOT CONSIDER DEPTH NOW
                img_feat = self.img_fc(img_feat)
                #x_v = torch.cat((x_v, img_feat.unsqueeze(0).expand(-1, N_V, -1)), dim=-1) # The followings are the target feature
                x_v = torch.cat((x_v, img_feat.unsqueeze(1).expand(N_B, N_V, -1)), dim=-1) # The followings are the target feature

        # expresssion encoder
        pn_feat_return = self.encoder(x_v, batch_mass=batch_mass, batch_L_val=batch_L_val, batch_evals=batch_evals, batch_evecs=batch_evecs, batch_gradX=batch_gradX, batch_gradY=batch_gradY)
        
        # cahce the exp
        self.cached_exp = pn_feat_return
        return pn_feat_return

    def encode_exp(self, x_v):
        """
        args:
            x_v: [B x N_V x 6] -  vertex feature
        output:
            source_feat: [B x latent_shape]
        """
        return self.encode(x_v)

    def encode_iden(self, x_v):
        """
        args:
            x_v: [B x N_V x 6] -  vertex feature
        output:
            source_feat: [B x latent_shape]
        """

        if self.global_pn is not None:

            source_feat = self.global_pn(x_v)
            self.cached_iden = source_feat
            return source_feat
     
    def decode(self, x_f, latent, x_f_orig=None, x_v_orig=None, source_img=None):

        x_f, x_source_v = x_f
        N_F = x_f.shape[1]
        B_S = x_f.shape[0]
        # Iden enc
        if latent.shape[1] == self.latent_shape : # Only the exp code is provided
            if self.global_pn and not self.img_warp:
                
                if x_v_orig is not None:
                    latent_iden= self.global_pn(x_v_orig)
                else:
                    latent_iden= self.global_pn(x_source_v)
                    
                latent_all = torch.cat((latent_iden, latent), axis=-1)
            else:
                latent_all = latent
                latent_iden = None
        else: # Fixed both exp and iden code
            latent_iden = latent[:, :-53]
            latent_all = latent
        latent_all = latent_all.unsqueeze(1).expand(B_S, N_F, -1)

        # MLP
        x = torch.cat([x_f, latent_all], dim=-1)
        out = x
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)

        out = self.linear_out(out)
        self.cached_feat = latent_all[:, 0]
        return out, latent_iden
        
    def sample(self, x_f):
        x_f = torch.cat([x_f, self.cached_feat.unsqueeze(1).expand(-1, x_f.shape[1], -1)], dim=-1)
        out = x_f
        for _ in range(len(self.linears)):
            out = torch.transpose(self.relu(self.gns[_](torch.transpose(self.linears[_](out), -1, -2))), -1, -2)

        out = self.linear_out(out)
        return out
    
class NFR(nn.Module):
    """ NFR architecture """
    def __init__(self, 
                 opts, 
                 pre_computes=None,
                 device='cpu',
                 mode='default'
                ):
        super(NFR, self).__init__()
        self.opts = opts
        self.mode=mode
        self.device=device
        
        self.is_train = self.opts.is_train if opts is not None else False
        
        self.ltn_dim = self.opts.ltn_dim if opts is not None else 128
        self.rig_dim = self.opts.rig_dim if opts is not None else 128
        self.id_dim = self.opts.id_dim if opts is not None else 128
        self.img_feat_dim = self.opts.img_feat_dim if opts is not None else 128
        
        self.in_key = self.opts.feature_type if opts is not None else 'cents&norms'
        self.out_key = self.opts.dec_type if opts is not None else 'disp'
        
        ### mesh autoencoder
        in_shape_dict = {'cents&norms':6, 'cents':3, 'cents&norms&seg':7}
        out_shape_dict = {'vert': 3, 'disp': 3, 'jacob': 9}
        
        self.img_encoder = TextureEncoder()
        self.img_fc = nn.Linear(128, self.img_feat_dim)
        
        self.id_encoder = BaseDiffusionNetEncoder(
            in_shape=in_shape_dict[self.in_key]+self.img_feat_dim,
            pre_computes=pre_computes,
            out_shape=self.id_dim,
        )
        self.exp_encoder = BaseDiffusionNetEncoder(
            in_shape=in_shape_dict[self.in_key]+self.img_feat_dim,
            pre_computes = pre_computes,
            out_shape=self.rig_dim,
        )
        self.decoder = BaseDecoder(
            #in_shape=3+3+128+100+53, ### NFR original setting
            in_dim=in_shape_dict[self.in_key]+self.img_feat_dim+self.rig_dim+self.id_dim, 
            out_shape=out_shape_dict[self.out_key]
        )
        
        # print all layer params number
        print("============< NFR >============")
        print(f"mode: {self.mode}")
        print(f"[img_encoder]: \t{self.count_parameters(self.img_encoder)}")
        print(f"[img_fc]: \t{self.count_parameters(self.img_fc)}")
        print(f"[id_encoder]: \t{self.count_parameters(self.id_encoder)}")
        print(f"[exp_encoder]: \t{self.count_parameters(self.exp_encoder)}")
        print(f"[decoder]: \t{self.count_parameters(self.decoder)}")
        print("-------------------------------")
        print(f"[total]: \t{self.count_parameters(self)}")
        print("===============================")
        
        self.criterion = nn.L1Loss()
        self.calc_norm_torch = calc_norm_torch
        
        if self.opts.dec_type=='jacob':
            self.normalizer = nfr_utils.Normalizer(self.opts.std_file, self.device)
            self.myfunc = deformation_gradient.apply
        
        if self.is_train:
            self.ict_face_model = ICT_face_model(face_only=False, device=self.device)
            self.ict_neutral = torch.from_numpy(self.ict_face_model.neutral_verts).to(self.device)
            self.ict_faces = torch.from_numpy(self.ict_face_model.faces).long().to(self.device)
            self.neutral_span_inv_matrix_np = np.linalg.inv(
                self.get_span_matrix_numpy(self.ict_neutral[None,None], self.ict_faces)
            )
        else:
            self.renderer       = nfr_utils.renderer(view_d=2.5, img_size=256, fragments=True)
            self.img_normalizer = nfr_utils.Normalizer_img('./data/MF_all_v5', self.device)

    def count_parameters(self, model):
        try:
            return sum(p.numel() for p in model if p.requires_grad)
        except:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_img_feat(self, img):
        return self.img_fc(self.img_encoder(img[..., :3].permute(0, 3, 1, 2)))
    
    def get_span_matrix_numpy(self, batch_time_v, faces):
        """
        Args
            batch_time_v (torch.tensor): [B, T, V, 3]
            faces (torch.tensor): [F, 3]
        Return
            V (np.array): [v2-v1, v3-v1, v4-v1]
        """
        B_v = batch_time_v.cpu().numpy()
        B_v = B_v.reshape([-1, B_v.shape[2], B_v.shape[3]])

        faces  = faces[None].repeat(B_v.shape[0], 1 ,1).cpu().numpy()
        B, num_faces  = faces.shape[:2]
        batch_indices = np.arange(B)[:, None, None]
        batch_indices = np.tile(batch_indices, (1, num_faces, 1))
        B_vf = B_v[batch_indices, faces].transpose(0, 1, 3, 2)

        v1 = B_vf[..., 0]
        v2 = B_vf[..., 1]
        v3 = B_vf[..., 2]

        cross = np.cross(v2 - v1, v3 - v1)
        v4 = v1 + cross / np.sqrt(np.linalg.norm(cross, axis=-1))[..., np.newaxis]

        return np.stack((v2 - v1, v3 - v1, v4 - v1), axis=-1)

    def get_jacobian_matrix_numpy(self, verts, faces, return_torch=False):
        B, T, FV, V = verts.shape
        
        span_matrix_np = self.get_span_matrix_numpy(verts, faces)
        Q_np = (span_matrix_np @ self.neutral_span_inv_matrix_np.repeat(T, axis=0)).transpose(0, 1, 3, 2)
        Q_np = Q_np.reshape([B, T, *Q_np.shape[1:]])

        if return_torch:
            return torch.tensor(Q_np)
        else:
            return Q_np
    
    def get_input_features(self, vertices, faces):
        """ for pretrained NFR model
        Args:
            vertices:       (torch.tensor) [V, 3] vertices of the mesh
            faces:          (torch.tensor) [F, 3] vertex indices in each face of the mesh
        Return:
            inputs:         (torch.tensor) [F, 6+128] per-face features
            input_source_v: (torch.tensor) [V, 6+128] per-vertex features
        """
        
        ## calc center of model
        mesh = nfr_utils.Mesh(
            vertices.detach().cpu().numpy(), 
            faces.detach().cpu().numpy(),
        )

        cents = nfr_utils.calc_cent(mesh.vertices, mesh.faces)
        cents = torch.from_numpy(cents).float().unsqueeze(0)

        ## normals
        _, norms = nfr_utils.calc_cent(mesh.vertices, mesh.faces)
        norms = torch.from_numpy(norms).float().unsqueeze(0)

        # set inputs [B, F, 6]
        inputs = torch.cat([cents, norms], dim=-1)
        inputs = inputs.to('cuda')

        #norms_v = torch.from_numpy(igl.per_vertex_normals(mesh.vertices, mesh.faces)).float()

        #f torch.isnan(norms_v).any():
        # If something wrong with the igl computation
        norms_v, _ = nfr_utils.calc_norm(mesh)
        #norms_v = calc_norm_torch(norms_v[None], faces, at='vert')
        norms_v = torch.from_numpy(norms_v).float()

        # set source vertex [B, V, 6]
        input_source_v = torch.cat([torch.from_numpy(mesh.vertices), norms_v], dim=-1).float().unsqueeze(0)
        input_source_v = input_source_v.to('cuda')

        # get image feture [B, 128]
        img = self.renderer.render_img(vertices, faces, self.renderer)        
        img_feat = self.get_img_feat(img)

        # set inputs [B, F, 6+128]
        inputs = torch.cat([inputs, img_feat.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)

        # if not args.img_only_mlp:
        input_source_v = torch.cat([input_source_v, img_feat.unsqueeze(1).expand(-1, input_source_v.shape[1], -1)], dim=-1)

        return inputs, input_source_v
    
    def calc_vert(self, g_pred, myfunc, operators):
        """Poisson solve
        Args:
            g_pred (torch.tensor): jacobian of the mesh
            myfunc (torch.autograd.Function): deformation_gradient
            operators (tuple): tuple of mesh operators for poisson solving (SuperLU, face_to_vert_idxs, face_to_vert_values, cupy csr matrix)
        Return:
            out_pred (torch.tensor): deformed vertices of the mesh
        """
        lu_solver, idxs, vals, rhs = operators
        out_pred = myfunc(g_pred, lu_solver, idxs, vals, rhs.shape)
        out_pred = out_pred - out_pred.mean(axis=[0, 1], keepdim=True)
        return out_pred