import os
import glob


"""
bash file to run this script
python rename_arkit_file.py --arkit_dir /data/ICT-audio2face/raw/;

"""

def main(arkit_dir="/data/ICT-audio2face/raw/"):
    tmp = glob.glob(os.path.join(arkit_dir,'*'))
    ids = [x for x in tmp if os.path.isdir(x)]
    ids.sort()
    print(ids)
    for id_ in ids:
        tmp = glob.glob(os.path.join(id_, "*"))
        audio_ids = [x for x in tmp if os.path.isdir(x)]
        audio_ids.sort()
        # import pdb;pdb.set_trace()
        for index, audio_id in enumerate(audio_ids):
            basename,_ = os.path.split(audio_id)
            save_fn = os.path.join(basename, f"{index:03d}")
            print(audio_id, save_fn)
            os.rename(audio_id, save_fn)


        
if __name__ == "__main__":
    import fire
    fire.Fire(main)

