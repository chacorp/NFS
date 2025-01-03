import torchaudio
import torch
import numpy as np


class AudioHandler:
    def __init__(self, device="cuda:0", model_type="wav2vec2", is_base=True):

        """
        model_type: https://pytorch.org/audio/stable/pipelines.html

        """
        if model_type == "wav2vec2":
            if is_base:
                self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            else:
                self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        elif model_type == "hubert":
            if is_base:
                self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
            else:
                self.bundle = torchaudio.pipelines.HUBERT_ASR_XLARGE
        elif model_type == "wavlm":
            self.bundle = torchaudio.pipelines.WAVLM_LARGE
        
        print("Sample Rate:", self.bundle.sample_rate)
        print("Labels:", self.bundle.get_labels())
        self.device = device
        self.model = self.bundle.get_model().to(self.device)

    def inference(self, wav_file, vid_len=None, target_fps=30, no_interp=False):
        waveform, sample_rate = torchaudio.load(wav_file)
        waveform = waveform.to(self.device)
        waveform = waveform[0:1,...]
        if sample_rate != self.bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.bundle.sample_rate)
        
        # extract acoustic features
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)

        # feature classification
        with torch.inference_mode():
            torch_emission, _ = self.model(waveform)
        
        emission = torch_emission[0].cpu().numpy()

        if no_interp:
            return features, emission
        else:
            # Resample network output from 50 fps to target fps
            INPUT_RATE = 50
            emission = self.interpolate_features(emission, INPUT_RATE, target_fps, output_len=vid_len)

            out_features = []
            for idx, feature in enumerate(features):
                if torch.is_tensor(feature):
                    np_feat = feature.cpu().numpy()[0]
                np_tmp = self.interpolate_features(np_feat, INPUT_RATE, target_fps, output_len=vid_len)
                out_features.append(np_tmp)
            return out_features, emission

    @classmethod
    def interpolate_features(cls, features, input_rate, output_rate, output_len=None):
        num_features = features.shape[1]
        input_len = features.shape[0]
        seq_len = input_len / float(input_rate)
        if output_len is None:
            output_len = int(seq_len * output_rate)
        input_timestamps = np.arange(input_len) / float(input_rate)
        output_timestamps = np.arange(output_len) / float(output_rate)
        output_features = np.zeros((output_len, num_features))
        for feat in range(num_features):
            output_features[:, feat] = np.interp(output_timestamps, input_timestamps, features[:, feat])
        return output_features



"""


for id in {00..09}; do 
    echo $id
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/m$id --model_type hubert;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/w$id --model_type hubert;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/m$id --model_type wav2vec2;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/w$id --model_type wav2vec2;
done


for id in {00..09}; do 
    echo $id
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/m$id --model_type wavlm;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/w$id --model_type wavlm;
done


for id in {00..09}; do 
    echo $id
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/m$id --model_type hubert;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/w$id --model_type hubert;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/m$id --model_type wav2vec2;
    python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/data_30fps/w$id --model_type wav2vec2;
done



# 2024/07/30 -------------------
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/train/ --model_type hubert --target_fps 25 --data_type biwi
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/val/ --model_type hubert --target_fps 25 --data_type biwi
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/test/ --model_type hubert --target_fps 25 --data_type biwi

python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/VOCASET/original_set/train --model_type hubert --target_fps 30 --data_type vocaset
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/VOCASET/original_set/test --model_type hubert --target_fps 30 --data_type vocaset
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/VOCASET/original_set/val --model_type hubert --target_fps 30 --data_type vocaset

python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/multiface/audio2face/ --model_type hubert --target_fps 30 --data_type mf

python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/ICT-audio2face/split_set/train --model_type hubert --target_fps 30 --data_type ict

# 2024/08/12 -------------------
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/test/ --model_type wav2vec2 --target_fps 25 --data_type biwi
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/val/ --model_type wav2vec2 --target_fps 25 --data_type biwi
python data_capture/audio_preprocess_wave2vec2.py --base_dir /data/BIWI/train/ --model_type wav2vec2 --target_fps 25 --data_type biwi
"""
if __name__ == "__main__":
    
    import os, glob
    import argparse

    from tqdm import tqdm
    import mediapy as mp
    import torch
    # load audio
    import natsort
    
    ## for BIWI ---------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Image projector to the generator latent spaces")
    parser.add_argument("--base_dir", type=str, default="/data/ICT-audio2face/data/valid_30fps/m02")
    parser.add_argument("--model_type", type=str, default="hubert") # hubert, wav2vec2
    parser.add_argument("--target_fps", type=int, default=30)
    parser.add_argument("--data_type", type=str, default="ict")
    opts = parser.parse_args()
    
    # os.makedirs(os.path.join(opts.base_dir, "wav2vec"), exist_ok=True)
    audio_handler = AudioHandler(model_type=opts.model_type)
    
    if opts.data_type == 'biwi' or opts.data_type == 'mf':
        AUDIO_DIR = os.path.join(opts.base_dir, "wav")
        
        if opts.data_type == 'ict':
            V_DIR = os.path.join(opts.base_dir, "csv")
        else:
            V_DIR = os.path.join(opts.base_dir, "vertices_npy")
            
        WV_DIR = os.path.join(opts.base_dir, f"{opts.model_type}")
        os.makedirs(WV_DIR, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(AUDIO_DIR, '*.wav'))
        wav_files = natsort.natsorted(wav_files)
        print(f"len of wav: {len(wav_files)}")

        pbar = tqdm(enumerate(wav_files))
        for idx, wav_file in pbar:
            id_sent = wav_file.split('/')[-1].split('.')[0]
            
            if not os.path.exists(f'{V_DIR}/{id_sent}.npy'):
                continue
            npy_file = f'{V_DIR}/{id_sent}.npy'
            npy_ = np.load(npy_file)
            vid_len = npy_.shape[0]
            
            if opts.data_type == 'vocaset':
                vid_len = npy_[::2].shape[0]
            
            # get ds features
            #import pdb;pdb.set_trace()
            features, logits = audio_handler.inference(wav_file, vid_len=vid_len, target_fps=opts.target_fps, no_interp=False)
            #features, logits = audio_handler.inference(wav_file, target_fps=opts.target_fps, no_interp=False)

            wav_name = os.path.basename(wav_file)
            # print(f"[{wav_name:<6}] len of vert anime:", vid_len)
            # print(f"[{wav_name:<6}] len of audio feat:", len(logits))
            # print(f"[{wav_name:<6}] audio feat shape:",  logits.shape)
            # for idx, feature in enumerate(features):
            #     print(f"[{wav_name:<6}] audio {idx:02d} feat shape:", feature.shape)
            # print("------------------------------")
            pbar.set_description(f"[{wav_name:<6}] {npy_.shape} | {logits.shape} | {features[0].shape}")

            basename = wav_name.split(".")[0]
            logit_path = os.path.join(WV_DIR, f"logits")
            os.makedirs(logit_path, exist_ok=True)
            np.save(os.path.join(logit_path, f"{basename}.npy"), logits)

            for idx, feature in enumerate(features):
                if torch.is_tensor(feature):
                    np_feat = feature.cpu().numpy()[0]
                else:
                    np_feat = feature
                os.makedirs(os.path.join(WV_DIR,f"{idx:02d}"), exist_ok=True)
                np.save(os.path.join(WV_DIR,f"{idx:02d}", f"{basename}.npy"), np_feat)
    elif opts.data_type == 'vocaset' or opts.data_type == 'ict':
        #import pdb;pdb.set_trace()
        ID_DIR = sorted(glob.glob(os.path.join(opts.base_dir, "*")))
        
        for ID_dir in ID_DIR:
            if opts.data_type == 'vocaset':
                V_DIR = os.path.join(ID_dir, "vertices_npy")
            else:
                V_DIR = os.path.join(ID_dir, "rig_param")
            
            WV_DIR = os.path.join(ID_dir, f"{opts.model_type}")
            os.makedirs(WV_DIR, exist_ok=True)
            
            wav_files = glob.glob(os.path.join(ID_dir, 'wav', '*.wav'))
            wav_files = natsort.natsorted(wav_files)
            print(f"len of wav: {len(wav_files)}")
            
            #import pdb;pdb.set_trace()
            pbar = tqdm(enumerate(wav_files))
            for idx, wav_file in pbar:
                id_sent = wav_file.split('/')[-1].split('.')[0]
                
                v_npy = f'{V_DIR}/{id_sent}.npy'
                    
                if not os.path.exists(v_npy):
                    continue
                #import pdb;pdb.set_trace()
                npy_ = np.load(v_npy)
                
                if opts.data_type == 'vocaset':
                    vid_len = npy_[::2].shape[0]
                else:
                    vid_len = npy_.shape[0]
                
                # get ds features
                features, logits = audio_handler.inference(wav_file, vid_len=vid_len, target_fps=opts.target_fps, no_interp=False)

                wav_name = os.path.basename(wav_file)
                # print(f"[{wav_name:<6}] len of vert anime:", vid_len)
                # print(f"[{wav_name:<6}] len of audio feat:", len(logits))
                # print(f"[{wav_name:<6}] audio feat shape:",  logits.shape)
                # for idx, feature in enumerate(features):
                #     print(f"[{wav_name:<6}] audio {idx:02d} feat shape:", feature.shape)
                # print("------------------------------")
                pbar.set_description(f"[{wav_name:<6}] {vid_len:^5} | {logits.shape} | {features[0].shape}")

                basename = wav_name.split(".")[0]
                logit_path = os.path.join(WV_DIR, f"logits")
                os.makedirs(logit_path, exist_ok=True)
                np.save(os.path.join(logit_path, f"{basename}.npy"), logits)

                for idx, feature in enumerate(features):
                    if torch.is_tensor(feature):
                        np_feat = feature.cpu().numpy()[0]
                    else:
                        np_feat = feature
                    os.makedirs(os.path.join(WV_DIR,f"{idx:02d}"), exist_ok=True)
                    np.save(os.path.join(WV_DIR,f"{idx:02d}", f"{basename}.npy"), np_feat)
    ## ------------------------------------------------------------------------------------------------------------
            
#     import os, glob
#     import argparse
#     parser = argparse.ArgumentParser(description="Image projector to the generator latent spaces")
#     parser.add_argument("--base_dir", type=str, default="/data/ICT-audio2face/data/valid_30fps/m02")
#     parser.add_argument("--model_type", type=str, default="hubert")
#     parser.add_argument("--target_fps", type=int, default=30)
#     opts = parser.parse_args()
    
#     # os.makedirs(os.path.join(opts.base_dir, "wav2vec"), exist_ok=True)
#     AUDIO_DIR = os.path.join(opts.base_dir, "wav")
#     CSV_DIR = os.path.join(opts.base_dir, "csv")
#     WV_DIR = os.path.join(opts.base_dir, f"{opts.model_type}")
#     os.makedirs(WV_DIR, exist_ok=True)
#     # load audio
#     import natsort
#     wav_files = glob.glob(os.path.join(AUDIO_DIR, '*.wav'))
#     wav_files = natsort.natsorted(wav_files)
#     print(f"len of wav: {len(wav_files)}")
#     csv_files = glob.glob(os.path.join(CSV_DIR, '*.csv'))
#     csv_files = natsort.natsorted(csv_files)
#     print(f"len of csv: {len(csv_files)}")

    

#     from tqdm import tqdm
#     import mediapy as mp
#     import torch
#     assert len(wav_files) ==len(csv_files), "check the file numbers"
#     # set hanlder
#     audio_handler = AudioHandler(model_type=opts.model_type)

#     import csv
#     def read_csv(f):
#         with open(f, 'r') as csv_file:
#             csv_reader = csv.DictReader(csv_file)
#             data = [row for row in csv_reader]
#         return data # [dict, dict, ...] len(data) = number of frames

#     # for idx, (wav_file, vid_file, ws_file) in tqdm(enumerate(zip(wav_files, vid_files, ws_files))):
#     for idx, (wav_file, csv_file) in tqdm(enumerate(zip(wav_files, csv_files))):
#         print(wav_file, csv_file)
#         vid_len = len(read_csv(csv_file))
        
#         # get ds features
#         features, logits = audio_handler.inference(wav_file, vid_len, target_fps=opts.target_fps)

#         wav_name = os.path.basename(wav_file)
#         # print(f"[{vid_name:<24}] len of video:", vid_len)
#         print(f"[{wav_name:<6}] len of audio feat:", len(logits), vid_len)
#         print(f"[{wav_name:<6}] audio feat shape:", logits.shape)
#         for idx, feature in enumerate(features):
#             print(f"[{wav_name:<6}] audio feat shape:", feature.shape)
#         print("------------------------------")

#         basename = wav_name.split(".")[0]
#         os.makedirs(os.path.join(WV_DIR, f"logits"), exist_ok=True)
#         np.save(os.path.join(WV_DIR, "logits", f"{basename}.npy"), logits)

#         for idx, feature in enumerate(features):
#             if torch.is_tensor(feature):
#                 np_feat = feature.cpu().numpy()[0]
#             else:
#                 np_feat = feature
#             os.makedirs(os.path.join(WV_DIR,f"{idx:02d}"), exist_ok=True)
#             np.save(os.path.join(WV_DIR,f"{idx:02d}", f"{basename}.npy"), np_feat)
