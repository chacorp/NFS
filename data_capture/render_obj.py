import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial
import numpy as np

import openmesh as om
#import vedo; vedo.settings.default_backend= "vtk"
import subprocess
import os

from tqdm import tqdm
import glob
import trimesh

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c,-s, 0, 0],
                      [ s, c, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ 1, 0, 0, 0],
                      [ 0, c,-s, 0],
                      [ 0, s, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(V, MVP, F, norm=True, z_div=True):
    if norm:
        V = (V - (V.max(0) + V.min(0)) *0.5) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    V = V @ MVP.T
    if z_div:
        V /= V[:, 3].reshape(-1, 1)
    VF = V[F]
    return VF

def calc_face_norm(V, F):
    fv = V[F]
    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)
    return norm

# def main(basedir, audio_fn, fps=60, norm=True):
#     ## visualize
#     basename = basedir
#     # print(basename)
#     meshes = sorted(glob.glob(os.path.join(basename, '*.obj')))
#     len_m = len(meshes)
#     print(basename, len_m)

#     ## visualize
#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)


#     ## initial frame
#     mesh = trimesh.load(meshes[0])

#     V = mesh.vertices
#     F = mesh.faces
    
#     ## normalize
#     if norm:
#         V = (V-(V.max(0)+V.min(0))/2)/max(V.max(0)-V.min(0))

#     ## homogeneous
#     V = np.c_[V, np.ones(len(V))]

#     ## MVP
#     model = translate(0, 0, -2.5) @ yrotate(-20)
#     proj  = perspective(30, 1, 1, 100)
#     MVP   = proj @ model # view is identity

#     V = V @ MVP.T
#     V /= V[:, 3].reshape(-1,1)

#     VF = V[F]
#     T  =  VF[:,:,:2]

#     Z = -VF[:,:,2].mean(axis=1)
#     zmin, zmax = Z.min(), Z.max()
#     Z = (Z-zmin)/(zmax-zmin)

#     C = plt.get_cmap("gray")(Z)
#     I = np.argsort(Z)
#     T, C = T[I,:], C[I,:]

#     collection = PolyCollection(T, closed=True, linewidth=0.1, facecolor=C, edgecolor="black")
#     ax.add_collection(collection)


#     ## render all frames
#     def update(i, meshes, MVP, F, col):
#         mesh = trimesh.load(meshes[i])
#         V = mesh.vertices
#         V = (V-(V.max(0)+V.min(0))/2)/max(V.max(0)-V.min(0))

#         ## homogeneous
#         V = np.c_[V, np.ones(len(V))]

#         V = V @ MVP.T
#         V /= V[:,3].reshape(-1,1)

#         VF= V[F]
#         T =  VF[:,:,:2]    
#         Z = -VF[:,:,2].mean(axis=1)
        
#         zmin, zmax = Z.min(), Z.max()
#         Z = (Z-zmin)/(zmax-zmin)
        
#         I = np.argsort(Z)
#         T = T[I,:]

#         # progress bar
#         print('\r{:04d}: {}'.format(i+1, '|'*(i%10) ), end="")
        
#         col.set_paths(T)
#         return fig,
    
#     ## save animation
#     anim = FuncAnimation(fig, partial(update, meshes=meshes, MVP=MVP, F=F, col=collection), frames=len_m)
#     anim.save(f'{basename}/tmp.mp4', fps=fps)


    
#     # mux audio and video
#     cmd = f"ffmpeg -y -i {audio_fn} -i {basename}/tmp.mp4 -c:v copy -c:a aac {basename}/out.mp4"
#     subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

#     # remove tmp files
#     subprocess.call(f"rm -f {basename}/tmp.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    
# def render(basedir="tmp",
#            savedir="tmp",
#            save_name="tmp",
#            figsize=(2,2),
#            fps=30,
#           ):
#     # make dirs
#     os.makedirs(savedir, exist_ok=True)
#     # if file exist
#     if os.path.isfile(f"{savedir}/tmp.mp4"):
#         print(f"{savedir}/tmp.mp4 already exist")
#         return
    
#     ## visualize
    
#     ## visualize
#     fig = plt.figure(figsize=figsize)
#     fig_xlim = [-0.5*figsize[0],+0.5*figsize[0]]
#     fig_ylim = [-0.5*figsize[1],+0.5*figsize[1]]
#     ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
    
#     meshes = sorted(glob.glob(os.path.join(basedir, '*.obj')))
#     tmp = trimesh.load(meshes[0])
#     F = tmp.faces
    
#     Vs = []
#     for mesh in meshes:
#         tmp = trimesh.load(mesh)
#         Vs.append(tmp.vertices)
#     Vs = np.array(Vs)

#     ## MVP
#     model = translate(0, 0, -2.5) @ yrotate(-20)
#     proj  = perspective(25, 1, 1, 100)
#     MVP   = proj @ model # view is identity

#     def render_mesh(ax, V, MVP, F):
#         # quad to triangle    
#         VF_tri = transform_vertices(V, MVP, F)

#         T = VF_tri[:, :, :2]
#         Z = -VF_tri[:, :, 2].mean(axis=1)
#         zmin, zmax = Z.min(), Z.max()
#         Z = (Z - zmin) / (zmax - zmin)

#         C = plt.get_cmap("gray")(Z)
#         I = np.argsort(Z)
#         T, C = T[I, :], C[I, :]

#         collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
#         ax.add_collection(collection)

#     def update(V):
#         # Cleanup previous collections
#         for coll in ax.collections:
#             coll.remove()

#         # Render meshes for all views
#         render_mesh(ax, V, MVP, F)

#         return ax.collections

#     anim = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering objs", ncols=100), blit=True)
#     anim.save(f'{savedir}/{save_name}.mp4', fps=fps)


# def render_vf(
#         Vs, 
#         F,
#         savedir="tmp",
#         save_name="tmp",
#         fps=30,
#         figsize=(2,2),
#         xrot=0,
#         yrot=-20,
#         zrot=0,
#         norm=False,
#     ):
#     """
#     Args:
#         Vs (torch.tensor): Batched mesh vertices
#         F (torch.tensor):  Face indcies of the mesh
#     """
#     # make dirs
#     os.makedirs(savedir, exist_ok=True)
        
#     ## visualize
#     fig = plt.figure(figsize=figsize)
#     fig_xlim = [-0.5*figsize[0],+0.5*figsize[0]]
#     fig_ylim = [-0.5*figsize[1],+0.5*figsize[1]]
#     ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
    
#     Vs = np.array(Vs.detach().cpu())
#     F  = np.array(F.detach().cpu())
    
#     ## MVP
#     model = translate(0, 0, -2.5) @ yrotate(yrot) @ xrotate(xrot) @ zrotate(zrot)
#     proj  = perspective(30, 1, 1, 100)
#     MVP   = proj @ model # view is identity

#     def render_mesh(ax, V, MVP, F):
#         # quad to triangle    
#         VF_tri = transform_vertices(V, MVP, F)

#         T = VF_tri[:, :, :2]
#         Z = -VF_tri[:, :, 2].mean(axis=1)
#         zmin, zmax = Z.min(), Z.max()
#         Z = (Z - zmin) / (zmax - zmin)

#         C = plt.get_cmap("gray")(Z)
#         I = np.argsort(Z)
#         T, C = T[I, :], C[I, :]

#         collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
#         ax.add_collection(collection)

#     def update(V):
#         # Cleanup previous collections
#         for coll in ax.collections:
#             coll.remove()

#         # Render meshes for all views
#         render_mesh(ax, V, MVP, F)

#         return ax.collections

#     anim = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering objs", ncols=100), blit=True)
#     anim.save(f'{savedir}/{save_name}.mp4', fps=fps)   



def render_w_audio(basedir="tmp",
                   savedir="tmp",
                   audio_fn="tmp",
                   figsize=(3,4),
                   fps=30,
                   y_rot=-25,
                   light_dir=np.array([0,0,1]),
                   mode='mesh', 
                   linewidth=1,
                  ):
    # make dirs
    os.makedirs(savedir, exist_ok=True)
    
    ## visualize
    basename = basedir
    
    meshes = sorted(glob.glob(os.path.join(basename, '*.obj')))
    tmp = trimesh.load(meshes[0])
    mesh_face = tmp.faces
    
    mesh_vtxs = []
    for mesh in meshes:
        tmp = trimesh.load(mesh)
        mesh_vtxs.append(tmp.vertices)
    mesh_vtxs = np.array(mesh_vtxs)
    
    num_meshes = len(mesh_vtxs)
    print(num_meshes)
    size = 4
    
    ## visualize
    fig = plt.figure(figsize=figsize)
    _r = figsize[0] / figsize[1]
    fig_xlim = [-_r, _r]
    fig_ylim = [-1, +1]
    ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
    

    ## MVP
    model = translate(0, 0, -2.5) @ yrotate(y_rot)
    proj  = perspective(25, 1, 1, 100)
    MVP   = proj @ model # view is identity

    def render_mesh(ax, V, MVP, F):        
        # quad to triangle    
        VF_tri = transform_vertices(V, MVP, F)

        T = VF_tri[:, :, :2]
        Z = -VF_tri[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode=='shade':
            C = calc_face_norm(V, F) @ model[:3,:3].T
            I = np.argsort(Z) # -----------------------> depth sorting
            T, C = T[I, :], C[I, :]

            NI = np.argwhere(C[:,2] > 0).squeeze() # --> culling w/ normal
            T, C = T[NI, :], C[NI, :]
            
            C = np.clip((C @ light_dir), 0, 1) # ------> cliping range 0 - 1
            C = C[:,np.newaxis].repeat(3, axis=-1)
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        ax.add_collection(collection)
    
    def update(V):
        # Cleanup previous collections
        for coll in ax.collections:
            coll.remove()

        # Render meshes for all views
        render_mesh(ax, V, MVP, mesh_face)
        
        return ax.collections
    
    plt.tight_layout()
    #tqdm(mesh_vtxs, desc="rnd", ncols=60)
    anim = FuncAnimation(fig, update, frames=mesh_vtxs, blit=True)
    
    bar = tqdm(total=num_meshes, desc="rendering")
    anim.save(
        f'{savedir}/tmp2.mp4', 
        fps=fps,
        progress_callback=lambda i, n: bar.update(1)
    )

    # mux audio and video
    print("[INFO] mux audio and video")
    cmd = f"ffmpeg -y -i {audio_fn} -i {savedir}/tmp2.mp4 -c:v copy -c:a aac {savedir}/out.mp4"
    subprocess.call(cmd, shell=True)
    print(f"{savedir}/out.mp4")

    # remove tmp files
    subprocess.call(f"rm -f {savedir}/tmp2.mp4", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# mesh_file = "/source/kseo/audio2face/NFR_pytorch/results_eccv/ict/conv-wav2vec-04/mesh/m02_060/000.obj"
# tmp = trimesh.load(mesh_file, process=False, maintain_order=True)

# basedir= "/source/kseo/audio2face/NFR_pytorch/results_eccv/ict/conv-wav2vec-04/mesh/m00_060/"
# audio_fn= "/source/kseo/audio2face/NFR_pytorch/results_eccv/ict/conv-wav2vec-04/wav/m00_060.wav"
# render_w_audio(basedir, 'tmp', audio_fn)




if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    import fire
    fire.Fire(render_w_audio)

"""
for ID in m03 w02 w03 w04 w05 w06 w07
do
echo $ID
python data_capture/render_obj.py \
    --basedir /source/kseo/audio2face/NFR_pytorch/results_eccv/ict/faceformer/mesh/m00_060 \
    --audio_fn /source/kseo/audio2face/NFR_pytorch/results_eccv/ict/faceformer/wav/m00_060.wav \
    --savedir . \
    --fps 30
    
done

python data_capture/render_obj.py \
--basedir /source/kseo/audio2face/NFR_pytorch/results_eccv/ict/conv-wav2vec-04/mesh/m02_060 \
--audio_fn /source/kseo/audio2face/NFR_pytorch/results_eccv/ict/conv-wav2vec-04/wav/m02_060.wav \
--savedir . \
--fps 30
"""

# python data_capture/render_obj.py --basedir '/source/sihun/MAASA/results/voca3/source' --savedir 'source/sihun/MAASA/results/voca3/source' --fps '30'
