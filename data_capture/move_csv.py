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
         fn = "out_30fps.csv"
        ):
    
    has_ICT_orig = True
    os.makedirs(save_dir, exist_ok=True)
    
    
    file_paths = glob.glob(os.path.join(basedir,"*","*",fn))
    # file_paths = glob.glob(os.path.join(basedir,"m01","*",'out.csv'))
    # file_paths.sort()
    import natsort
    file_paths = natsort.natsorted(file_paths)
    print(file_paths[0:10])

    for idx, file_path in enumerate(file_paths):
        basepath, _ = os.path.split(file_path)
        tmp, audio_id = os.path.split(basepath)
        _, person_id = os.path.split(tmp)
        
        save_fn = f'{save_dir}/{person_id}/csv/{audio_id}.csv'
        save_dir_ = f'{save_dir}/{person_id}/csv/'
        print(file_path, save_fn)
        if os.path.isfile(save_fn):
            continue
        os.makedirs(save_dir_, exist_ok=True)
        shutil.copy(file_path, save_fn)
        
if __name__ == "__main__":
    import fire
    fire.Fire(main)



