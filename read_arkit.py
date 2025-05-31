##########################################################################################
#                                                                                        #
# ICT FaceKit                                                                            #
#                                                                                        #
# Copyright (c) 2020 USC Institute for Creative Technologies                             #
#                                                                                        #
# Permission is hereby granted, free of charge, to any person obtaining a copy           #
# of this software and associated documentation files (the "Software"), to deal          #
# in the Software without restriction, including without limitation the rights           #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell              #
# copies of the Software, and to permit persons to whom the Software is                  #
# furnished to do so, subject to the following conditions:                               #
#                                                                                        #
# The above copyright notice and this permission notice shall be included in all         #
# copies or substantial portions of the Software.                                        #
#                                                                                        #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR             #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,               #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE            #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                 #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,          #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE          #
# SOFTWARE.                                                                              #
##########################################################################################


import csv
import numpy as np
import os
from tqdm import tqdm

import sys

sys.path.append('./third_party/ICT-FaceKit/Scripts')
try:
    import face_model_io
    has_ICT_orig = True

except ModuleNotFoundError:
    has_ICT_orig = False

    
def transform_string(s):
    s = s.replace('_L', 'Left')
    s = s.replace('_R', 'Right')
    return s[0].upper() + s[1:]

def main(csv_file_path, out_path="output"):
    
    # Create a new FaceModel and load the model
    face_model = face_model_io.load_face_model('./third_party/ICT-FaceKit/FaceXModel')
    
    # Load blendshape names of ict-facekit
    ict_keys = [transform_string(item) for item in face_model._expression_names]
    
    # Create a output path
    os.makedirs(out_path, exist_ok=True)
    
    # Read a CSV file 
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]
        
    # Load blendshape names of ARkit
    data_keys = list(data[0].keys())
    
    # Get corresponding keys: sort ARkit BS based on ict BS
    key_list = []
    for key in ict_keys: 
        if key in data_keys:
            key_list.append(key)
        else:
            for _k in data_keys:
                if _k in key:
                    key_list.append(_k)
                    break
    
    # Export Mesh
    idx = 0
    for frame in tqdm(data):
        
        # blendshape 53
        ex_coeffs = [frame[key] for key in key_list]
        ex_coeffs = np.array(ex_coeffs, dtype=np.float64) # (53,)

        # Set blendshape coeffs
        face_model.set_expression(ex_coeffs)

        # Deform the mesh
        face_model.deform_mesh()

        # Write the deformed mesh
        face_model_io.write_deformed_mesh('{}/arkit_frame_{:06d}.obj'.format(out_path, idx), face_model)
        idx = idx + 1
        

if __name__ == '__main__':
    if not has_ICT_orig:
        logging.error('No face_model_io Module found!')
            
    # Example usage:
    csv_file_path = r"/data/sihun/arkit_CSH/MySlate_7_iPhone.csv"
    out_path = r"/data/sihun/arkit_CSH"
    
    main(csv_file_path, out_path)