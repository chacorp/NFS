import shutil
import sys
import csv
import numpy as np
import os
import glob
from tqdm import tqdm
import subprocess

def main(basedir = "/data/ICT-audio2face/data/train_30fps",
        savedir="/data/ICT-audio2face/data/valid_30fps",
        train_split=0.8):
    
    ids = glob.glob(os.path.join(basedir,"*"))
    ids.sort()
    print(ids[0:10])

    for idx, id_ in enumerate(ids):
        basepath, person_id = os.path.split(id_)
        
        print(basepath, person_id)        
        wav_fn = sorted(glob.glob(f'{id_}/wav/*.wav'))
        csv_fn = sorted(glob.glob(f'{id_}/csv/*.csv'))

        assert len(wav_fn) == len(csv_fn), "len not math"

        # split
        l = len(wav_fn)
        split_idx = int(l*train_split)
        wav_fn_train = wav_fn[:split_idx]
        csv_fn_train = csv_fn[:split_idx]
        wav_fn_valid = wav_fn[split_idx:]
        csv_fn_valid = csv_fn[split_idx:]

    
        for (wav_f, csv_f) in zip(wav_fn_valid, csv_fn_valid):
            print(wav_f, csv_f)
            # move
            save_wav_fn = wav_f.replace(basedir, savedir)
            shutil.move(wav_f, save_wav_fn)
            save_csv_fn = csv_f.replace(basedir, savedir)
            shutil.move(csv_f, save_csv_fn)
            print(save_wav_fn, save_csv_fn)
            print("-----")
        
if __name__ == "__main__":
    import fire
    fire.Fire(main)



