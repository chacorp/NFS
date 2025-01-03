import sys
import csv
import numpy as np
import os
import glob
from tqdm import tqdm
import subprocess

def main(basedir = '/data/ICT-audio2face/raw',
        #  save_path = "/data/ICT-audio2face/data",
        ):
    
    has_ICT_orig = True
    
    
    mp4_file_paths = glob.glob(os.path.join(basedir,"*","*",'*iPhone.mov'))
    mp4_file_paths.sort()
    print(mp4_file_paths[0:10])


    for idx, mp4_file_path in enumerate(mp4_file_paths):
        print(mp4_file_path)
        basepath, _ = os.path.split(mp4_file_path)
        
        save_fn = f"{basepath}/out.mp4"
        # if os.path.isfile(save_fn):
        #     continue
        cmd = f'ffmpeg -y -i {mp4_file_path} -vf "trim=2,setpts=PTS-STARTPTS" -af "atrim=2,asetpts=PTS-STARTPTS" {save_fn}'
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        save_audio_fn = f"{basepath}/out.wav"
        # if os.path.isfile(save_audio_fn):
        #     continue
        cmd_audio = f'ffmpeg -y -i {save_fn} -vn -vn -acodec pcm_s16le -ar 44100 -ac 2 {save_audio_fn}'
        subprocess.call(cmd_audio, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # Write the deformed mesh

        
if __name__ == "__main__":
    import fire
    fire.Fire(main)


# ffmpeg -y -i tmp.mp4 -vf "trim=end=1,setpts=PTS-STARTPTS" -af "atrim=end=1,asetpts=PTS-STARTPTS" output.mp4
# ffmpeg -sseof -7 -i tmp.mp4 -c copy output.mp4
# ffmpeg -y -sseof -1 -i tmp.mp4 -c copy output.mp4

ffmpeg -y -i MySlate_10_iPhone.mov -vf "trim=2,setpts=PTS-STARTPTS" -af "atrim=2,asetpts=PTS-STARTPTS" tmp.mp4
ffmpeg -y -i tmp.mp4 -vn -vn -acodec pcm_s16le -ar 44100 -ac 2 tmp.wav

duration=`ffprobe -v error -show_entries format=duration -of csv=p=0 tmp.mp4`

echo $duration

duration=$(bc <<< "$duration"-"1")

echo $duration

ffmpeg -y -ss 00:00:00 -to $duration -i tmp.mp4 -c copy out.mp4

ffmpeg -y -i out.mp4  -vn -vn -acodec pcm_s16le -ar 44100 -ac 2 out.wav