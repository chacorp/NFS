import os
import torch
import torch.nn as nn
import numpy as np
import subprocess
import trimesh
import time
from tqdm import tqdm
import mediapy as mp
import tempfile
from subprocess import call
os.environ['PYOPENGL_PLATFORM'] = 'egl' #'osmesa' # 
# os.environ['PYOPENGL_EGL_DEVICE_ID'] = 'egl'
import pyrender
try:
    import cv2
except:
    import os
    # os.sys.cmd("pip install opencv-python==4.5.5.64")
    exit(f"install opencv-python==4.5.5.64")


def render_mesh_helper(\
                       mesh,\
                       t_center, \
                       camera_params, \
                       rot=np.zeros(3), \
                       tex_img=None, \
                       z_offset=0, \
                       vertex_color=None,
                       H=800,
                       W=800):

    frustum = {'near': 0.01, 'far': 3.0, 'height': H, 'width': W}
    
    mesh_copy = trimesh.Trimesh(vertices=mesh.vertices - np.array([0, 0, 0.5]), faces=mesh.faces)
    mesh_copy.vertices[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.vertices-t_center).T).T+t_center
    # intensity = 2.0
    intensity = 1.0

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                metallicFactor=0.8, 
                roughnessFactor=0.8 
            )

    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.vertices, faces=mesh_copy.faces, vertex_colors=vertex_color)
    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=None, smooth=True)
    # render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

    if True:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    # lip_sm = trimesh.creation.uv_sphere(radius=0.001)
    # lip_sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    # lip_verts = verts[:,hp.lip_landmark,:]
    # lip_tfs = np.tile(np.eye(4), (lip_verts.shape[1], 1, 1))
    # lip_tfs[:,:3,3] = lip_verts[i]
    # lip_m = pyrender.Mesh.from_trimesh(lip_sm, poses=lip_tfs)
    # scene.add(lip_m)

    # exp_sm = trimesh.creation.uv_sphere(radius=0.001)
    # exp_sm.visual.vertex_colors = [0.0, 1.0, 1.0]
    # exp_verts = verts[:,hp.exp_landmark,:]
    # exp_tfs = np.tile(np.eye(4), (exp_verts.shape[1], 1, 1))
    # exp_tfs[:,:3,3] = exp_verts[i]
    # exp_m = pyrender.Mesh.from_trimesh(exp_sm, poses=exp_tfs)
    # scene.add(exp_m)

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    # import pdb; pdb.set_trace()
    scene.add(camera, pose=camera_pose)

#     angle = np.pi / 6.0
    angle = np.pi / 4.0
    
    pos = camera_pose[:3,3]
    light_color = np.array([1.0, 1.0, 1.0])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())
    
    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    # try:
    r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
    color, _ = r.render(scene, flags=flags)
    # except:
    #     print('pyrender: Failed rendering frame')
    #     color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def render_sequence(\
    verts, \
    output_path,\
    wav_path,\
    fn="sample",\
    mesh_type='voca',
    ):
    os.makedirs(output_path, exist_ok=True)
    print("rendering sequence...")
    print(f"\t[{fn}] {wav_path}")
    print(f"\t[save path] {output_path}")
    
    H, W = 800, 800
    # import pdb; pdb.set_trace()
    if mesh_type == 'voca':
        template_file="test-mesh/FLAME_sample.ply"
        camera_params = {'c': np.array([H//2, W//2]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif mesh_type == 'biwi':
        template_file="test-mesh/BIWI.ply"
        camera_params = {'c': np.array([H//2, W//2]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
    elif mesh_type == 'ict-full':
        template_file='/source/sihun/MAASA/ICT/precompute-fullhead/m00_mesh.obj'
        camera_params = {'c': np.array([H//2, W//2]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
    else: # ict-fo'
        template_file='/source/sihun/MAASA/ICT/precompute-face_only/m00_mesh.obj'
        camera_params = {'c': np.array([H//2, W//2]),
                        'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                        'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}
        
    # import pdb; pdb.set_trace()
    print(template_file)
    template = trimesh.load(template_file, maintain_order=True, process=False)
    
    
    predicted_vertices = verts
    num_frames = predicted_vertices.shape[0]
    
    center = np.mean(predicted_vertices[0], axis=0)

    # render video
    frames = []
    for i_frame in tqdm(range(num_frames)):
        #render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        render_mesh = trimesh.Trimesh(vertices=predicted_vertices[i_frame], faces=template.faces)
        pred_img = render_mesh_helper(render_mesh, center, camera_params, vertex_color=None, z_offset=-1.3, H=H,W=W)
        pred_img = pred_img.astype(np.uint8)
        frames.append(pred_img)
    frames = np.stack(frames, axis=0)
    
    # write
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    mp.write_video(f"{tmp_video_file.name}", frames, fps=30)
    
    # ffmpeg video
    video_fname = os.path.join(output_path, 'tmp.mp4')
    cmd = f'ffmpeg -y -i {tmp_video_file.name} -pix_fmt yuv420p -qscale 0 {video_fname}'
    call(cmd, shell=True)

    # mux audio video
    audio_fn = wav_path
    video_fn = video_fname
    new_video_fn = os.path.join(output_path, fn+'.mp4')
    cmd = f"ffmpeg -y -i {audio_fn} -i {video_fn} -c:v copy -c:a aac {new_video_fn}"
    call(cmd, shell=True)

    # remove
    os.remove(video_fname)