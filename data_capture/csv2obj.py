import sys
import csv
import numpy as np
import os
import glob
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")
sys.path.append('../third_party/ICT-FaceKit/Scripts')

import face_model_io
def transform_string(s):
    s = s.replace('_L', 'Left')
    s = s.replace('_R', 'Right')
    return s[0].upper() + s[1:]
    
def main(basedir = '/data/ICT-audio2face/data',
         save_path = "/data/ICT-audio2face/data",
         face_model = "../third_party/ICT-FaceKit/FaceXModel"
        ):
    
    has_ICT_orig = True
    # Create a new FaceModel and load the model
    face_model = face_model_io.load_face_model(face_model)
    ict_keys = [transform_string(item) for item in face_model._expression_names]
    print(ict_keys)
    
    os.makedirs(save_path, exist_ok=True)
    

    ids = glob.glob1(basedir,"*")
    ids.sort()
    print(ids[0:10])
    for person_id in ids[1:]:
        print(person_id)
        csv_file_paths = glob.glob(os.path.join(basedir,f"{person_id}","csv",'*.csv'))
        csv_file_paths.sort()
        print(csv_file_paths[0:10])


        for idx, csv_file_path in enumerate(csv_file_paths):
            basepath, audio_id = os.path.split(csv_file_path)
            audio_id = audio_id.split('.')[0]
            tmp, _ = os.path.split(basepath)
            # _, person_id = os.path.split(tmp)
            print(basepath, person_id, audio_id)

            # Converts a CSV file to a JSON file
            with open(csv_file_path, 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                data = [row for row in csv_reader]
            data_keys = list(data[0].keys())

            # ??????
            key_list = []
            for key in ict_keys: 
                if key in data_keys:
                    key_list.append(key)
                else:
                    for _k in data_keys:
                        if _k in key:
                            key_list.append(_k)
                            break

            os.makedirs(os.path.join(f'{save_path}',f'{person_id}'), exist_ok=True)
            os.makedirs(os.path.join(f'{save_path}',f'{person_id}/obj'), exist_ok=True)
            os.makedirs(os.path.join(f'{save_path}',f'{person_id}/obj/{audio_id}'), exist_ok=True)
            for i, frame in tqdm(enumerate(data)):
                save_fn = f'{save_path}/{person_id}/obj/{audio_id}/{i:06d}.obj'
                if os.path.isfile(save_fn):
                    continue
                ex_coeffs = []
                for key in key_list:
                    ex_coeffs.append(frame[key])

                # blendshape 49
                ex_coeffs = np.array(ex_coeffs, dtype=np.float64) # (49,)

                # Deform the mesh
                face_model.set_expression(ex_coeffs)
                face_model.deform_mesh()

                # Write the deformed mesh
                face_model_io.write_deformed_mesh(save_fn, face_model)
            
            # only the first
            break

if __name__ == "__main__":
    import fire
    fire.Fire(main)

"""
python csv2obj.py --basedir /data/ICT-audio2face/data_30fps --save_path ./obj --face_model ../third_party/ICT-FaceKit/FaceXModel

python render_obj.py --basedir /source/kseo/audio2face/NFR_pytorch/data_capture/obj/m05/obj/000 \
    --audio_fn /data/ICT-audio2face/data_30fps/m05/wav/000.wav


python render_obj.py --basedir /source/kseo/audio2face/NFR_pytorch/data_capture/obj/m06/obj/000 \
    --audio_fn /data/ICT-audio2face/data_30fps/m06/wav/000.wav --savedir m06

python render_obj.py --basedir /source/kseo/audio2face/NFR_pytorch/data_capture/obj/m04/obj/000 \
    --audio_fn /data/ICT-audio2face/data_30fps/m04/wav/000.wav --savedir m04

python render_obj.py --basedir /source/kseo/audio2face/NFR_pytorch/data_capture/obj/m03/obj/000 \
    --audio_fn /data/ICT-audio2face/data_30fps/m03/wav/000.wav --savedir m03

python render_obj.py --basedir /source/kseo/audio2face/NFR_pytorch/data_capture/obj/m02/obj/000 \
    --audio_fn /data/ICT-audio2face/data_30fps/m02/wav/000.wav --savedir m02



"""