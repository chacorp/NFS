import matplotlib.pyplot as plt
import glob
import os
import librosa
import librosa.display

import fire
import csv
import soundfile as sf
import json
import natsort
    

errors = []

def audio2frame(s, e, sr, fps):
    # s = 44100  # Example start sample
    # e = 88200    # Example end sample
    # sample_rate = 44100        # Example audio sample rate
    # frame_rate = 30            # Example video frame rate
    if fps == 30:
        print("\tFPS:", fps)
    start_frame = (s / sr) * fps
    end_frame = (e / sr) * fps
    l = (end_frame - start_frame) / fps

    print("\tStart Frame:", int(start_frame))
    print("\tEnd Frame:", int(end_frame))
    print(f"\tDuration: {l:3.3f}", "seconds")

    return int(start_frame), int(end_frame)


def trim_audio(y, sr=44100):
    
    tmp, splits = librosa.effects.trim(y, top_db=20)
    (s, e) = splits

    # Define the duration in seconds to trim (1 second)
    trim_duration = 1 # seconds
    # Calculate the number of frames to keep based on the desired duration
    pad = int(sr * trim_duration)
    # Trim the audio to keep the first 1 second
    y_trimmed = tmp[pad:-pad]
    (s, e) = (s+pad, e-pad)
    return y_trimmed, (s, e)

def write_csv(csv_fn, s, e):
    # Replace with the paths to your input and output files
    basedir, _ = os.path.split(csv_fn)
    new_fn = os.path.join(basedir, "out_30fps.csv")
    # Row numbers to split at
    start_row_number = s + 2  # Start row number
    end_row_number = e + 2  # End row number
    l = end_row_number - start_row_number + 1
    print("\tlen:", end_row_number - start_row_number + 1)

    # Open the input CSV file and create the output CSV files
    counter = 0
    with open(csv_fn, 'r') as input_file, \
            open(new_fn, 'w') as new_file:

        reader = csv.reader(input_file)
        new_writter = csv.writer(new_file)

        for row_number, row in enumerate(reader):
            if row_number == 0:
                # header
                new_writter.writerow(row)
            elif start_row_number <= row_number <= end_row_number:
                # split
                counter = counter + 1
                new_writter.writerow(row)

    print("\tcounter:", counter)
    # assert counter == l, f"counter != l, {counter} != {l}"
    global errors 
    if counter != l:
        print("counter != l, {counter} != {l}")
        errors += [csv_fn]
        # os.remove(new_fn)
        # raise Exception("counter != l, {counter} != {l}")

def main(basedir = '/data/ICT-audio2face/raw',
        ):
    
    ids = os.listdir(basedir)
    # filter
    m_ids = [os.path.join(basedir,id_) for id_ in ids if id_.startswith("m")]
    # m_ids = []  
    w_ids = [os.path.join(basedir,id_) for id_ in ids if id_.startswith("w")]

    ids = m_ids + w_ids
    ids.sort()
    print(ids)
    ids = natsort.natsorted(ids)
    for id_ in ids:

        # files
        audio_fns = glob.glob(os.path.join(id_,"*","*_iPhone.mov"))
        csv_fns = glob.glob(os.path.join(id_,"*","*_iPhone.csv"))
        csv_30fps_fns = glob.glob(os.path.join(id_,"*","_out_30fps.csv"))
        meta_audio_fns = glob.glob(os.path.join(id_,"*","audio_metadata.json"))
        meta_video_fns = glob.glob(os.path.join(id_,"*","take.json"))

        print(audio_fns)

        # sort
        audio_fns = natsort.natsorted(audio_fns)
        csv_fns = natsort.natsorted(csv_fns)
        csv_30fps_fns = natsort.natsorted(csv_30fps_fns)
        meta_audio_fns = natsort.natsorted(meta_audio_fns)
        meta_video_fns = natsort.natsorted(meta_video_fns)

        # audio_fns.sort()
        # csv_fns.sort()
        # meta_audio_fns.sort()
        # meta_video_fns.sort()

        # read json
        
        VIS = False
        for i in range(len(audio_fns)):
            with open(meta_audio_fns[i]) as f:
                meta_audio = json.load(f)
                sample_rate = meta_audio['SampleRate'] 

            with open(meta_video_fns[i]) as f:
                meta_video = json.load(f)
                fps = meta_video['videoTargetFrameRate']
                frame_len = meta_video['frames']
            if fps == 60:
                # half the fps
                frame_len = frame_len//2
                fps = fps//2
                csv_fn_ = csv_30fps_fns[i]
            elif fps == 30:
                csv_fn_ = csv_fns[i]
                pass
            
            basedir, _ = os.path.split(audio_fns[i])

            print(basedir, sample_rate, fps, frame_len)

            y, sr = librosa.load(audio_fns[i], sr=sample_rate)

            y_out, (s, e) = trim_audio(y, sr=sr)
            s_, e_ = audio2frame(s, e, sr=sr, fps=fps)

            write_csv(csv_fn_, s_, e_)
            if VIS:
                plt.figure(figsize=(12, 4))

                
                # Plot the waveform for each audio file
                librosa.display.waveshow(y, sr=sr, alpha=0.7)
                librosa.display.waveshow(y_out, sr=sr, alpha=0.7)

                # Set plot labels and title
                plt.title('Waveforms of Multiple Audio Files')
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
            
            # save audio
            sf.write(f'{basedir}/out.wav', y_out, sr, subtype='PCM_24')
            

            print(csv_fn_)


    print(errors)
    # write error files
    with open("errors.txt", "w") as f:
        for error in errors:
            f.write(error)
            f.write("\n")

if __name__ == "__main__":
    fire.Fire(main)