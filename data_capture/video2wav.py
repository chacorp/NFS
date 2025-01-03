import sys
import csv
import numpy as np
import os
import glob
from tqdm import tqdm
import subprocess

def main(basedir = '/data/ICT-audio2face/raw',
         save_path = "/data/ICT-audio2face/data",
        ):
    
    has_ICT_orig = True
    os.makedirs(save_path, exist_ok=True)
    
    
    mp4_file_paths = glob.glob(os.path.join(basedir,"*","*",'*iPhone.mov'))
    mp4_file_paths.sort()
    print(mp4_file_paths[0:10])

    for idx, mp4_file_path in enumerate(mp4_file_paths):
        basepath, _ = os.path.split(mp4_file_path)
        tmp, audio_id = os.path.split(basepath)
        _, person_id = os.path.split(tmp)
        
        print(basepath, person_id, audio_id)        
        save_fn = f'{save_path}/{person_id}/wav/{audio_id}.wav'
        save_dir = f'{save_path}/{person_id}/wav/'
        os.makedirs(save_dir, exist_ok=True)
        
        if os.path.isfile(save_fn):
            continue
        # Write the deformed mesh
        cmd = f"ffmpeg -i {mp4_file_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {save_fn}"
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        
if __name__ == "__main__":
    import fire
    fire.Fire(main)


