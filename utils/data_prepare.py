import argparse

import sys
from pathlib import Path
abs_path = str(Path(__file__).parents[1].absolute())
sys.path+=[abs_path]

from tqdm import tqdm
from utils import nfr_utils
from utils.mesh_utils import *
from utils.remesh_utils import map_vertices, decimate_mesh_vertex
import trimesh

# from .matplotlib_rnd import *
import glob

def load_mesh(mesh, renderer, device='cuda', process=True):
    if process:
        mesh = nfr_utils.get_biggest_connected(mesh)
        mesh = nfr_utils.remove_degenerated_triangles(mesh)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices.astype(np.float32)
    mesh_operators = get_mesh_operators(mesh)
    mesh_dfn_info = nfr_utils.get_dfn_info(mesh)
    mesh_dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in mesh_dfn_info]
    
    img = renderer.render_img(mesh).float().to(device)

    return mesh, mesh_operators, mesh_dfn_info, img


def Options():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--in_dir', default='', help='input path to mesh data (source neutral mesh)')
    parser.add_argument('-o', '--out_dir', default='', help='output path to save data')
    parser.add_argument("--print",    dest='print', action='store_true', help='if True, print details')
    parser.set_defaults(print=False)
    parser.add_argument("--decimate", dest='decimate', action='store_true', help='if True, apply decimate the given mesh')
    parser.set_defaults(print=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    """
    python data_prepare.py -i path_to_obj -o output_dir
    python utils/data_prepare.py -i /data/sihun/multiface_align/obj -o /data/sihun/multiface_align/precomputes
    python utils/data_prepare.py -i /data/sihun/ICT-audio2face/precompute-synth-narrow_face -o /data/sihun/ICT-audio2face/precompute-synth-narrow_face
    """

    args = Options()

    renderer = Renderer(view_d=2.5, img_size=256, fragments=True)

    src_mesh_list = glob.glob(os.path.join(args.in_dir,"*.obj"))
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for src_mesh_file in tqdm(src_mesh_list):
        dfn_info_filename = f'{args.out_dir}/{src_name}_dfn_info.pkl'
        operators_filename = f'{args.out_dir}/{src_name}_operators.pkl'
        img_filename = f'{args.out_dir}/{src_name}_img.npy'
        
        if os.path.isfile(dfn_info_filename):
            continue
        if os.path.isfile(operators_filename):
            continue
        if os.path.isfile(img_filename):
            continue
        
        src_name = src_mesh_file.split("/")[-1].replace(".obj", "")
        src_name = src_name.replace("_mesh", "")
            
        src_mesh = trimesh.load(src_mesh_file, maintain_order=True, process=False)
        if args.decimate:
            deci_mesh = decimate_mesh_vertex(src_mesh, num_vertex=4096, tolerance=0)
            _, v_map = map_vertices(src_mesh, deci_mesh)
            src_mesh.vertices = src_mesh.vertices[v_map]
            src_name = src_name+"_deci"

        src_mesh, src_mesh_operators, src_mesh_dfn_info, src_img = load_mesh(src_mesh, renderer)
        src_img = src_img.detach().cpu().numpy()
        
        N_FACE   = src_mesh.faces.shape[0]
        N_VERTEX = src_mesh.vertices.shape[0]

        if args.print:
            txt=f'ID: {src_name}\nProperties:\n'
            txt+=f'\tVertices:\t{N_VERTEX}\n'
            txt+=f'\tFaces:\t{N_FACE}\n'
            txt+=f'\tOperators\t{src_mesh_operators[0].shape}'
            print(txt)

        ## visualize mesh
        # v_list=[ src_mesh.vertices ]
        # f_list=[ src_mesh.faces ]
        # plot_image_array(v_list, f_list, rot_list=[[0,0,0]]*len(v_list), size=2, bg_black=False, mode='shade')
        
        with open(dfn_info_filename, mode='wb') as f:
            pickle.dump(src_mesh_dfn_info, f)
        with open(operators_filename, mode='wb') as f:
            pickle.dump(src_mesh_operators, f)
        np.save(img_filename, src_img)
        #_ = src_mesh.export(f'{args.out_dir}/{src_name}_mesh.obj')

    # save_mesh_list = glob(os.path.join(args.out_dir,"*_mesh.obj"))
    # assert len(src_mesh_list) == len(save_mesh_list), "mismatch between given mesh list and saved mesh list!"