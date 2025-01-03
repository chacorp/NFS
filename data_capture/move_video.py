import shutil
import sys
import csv
import numpy as np
import os
import glob
from tqdm import tqdm
import subprocess

def main(basedir = '/data/ICT-audio2face/raw',
         save_dir = "/data/ICT-audio2face/data",
        ):
    
    has_ICT_orig = True
    os.makedirs(save_dir, exist_ok=True)
    
    
    mp4_file_paths = glob.glob(os.path.join(basedir,"*","*",'*iPhone.mov'))
    mp4_file_paths.sort()
    print(mp4_file_paths[0:10])

    for idx, mp4_file_path in enumerate(mp4_file_paths):
        basepath, _ = os.path.split(mp4_file_path)
        tmp, audio_id = os.path.split(basepath)
        _, person_id = os.path.split(tmp)
        
        print(basepath, person_id, audio_id)        
        save_fn = f'{save_dir}/{person_id}/video/{audio_id}.mov'
        save_dir_ = f'{save_dir}/{person_id}/video/'
        if os.path.isfile(save_fn):
            continue
        os.makedirs(save_dir_, exist_ok=True)
        
        shutil.copy(mp4_file_path, save_fn)
        
if __name__ == "__main__":
    import fire
    fire.Fire(main)



