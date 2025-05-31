import csv
import bpy

# Get the object with the shape keys. Replace 'ObjectName' with the name of your object.
obj = bpy.data.objects['ICTFaceModel']


def transform_string_ICTFace_to_ARkit(s):
    s = s.replace('_L', 'Left')
    s = s.replace('_R', 'Right')
    return s[0].upper() + s[1:]

def transform_string_ARkit_to_ICTFace(s):
    s = s.replace('Left', '_L')
    s = s.replace('Right', '_R')
    return s[0].lower() + s[1:]

csv_file_path = r"C:\Users\sihun\Downloads\MySlate_10_iPhone.csv"

# """Converts a CSV file to a JSON file."""
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    
    data = [row for row in csv_reader]
data_keys = list(data[0].keys())
print(data_keys)


#FrameNumber = 0
for FrameNumber in range(len(data)):
    # Ensure the object has shape keys.
    if obj.data.shape_keys:

        # Reference to the shape key block
        shape_keys = obj.data.shape_keys.key_blocks
        
        for key in shape_keys:
            key_ar = transform_string_ICTFace_to_ARkit(key.name)
            if key_ar in data_keys:
                key.value = float(data[FrameNumber][key_ar])
                key.keyframe_insert(data_path="value", frame=FrameNumber)
            else:
                for _k in data_keys:
                    if _k in key.name:
                        key.value = float(data[FrameNumber][_k])
                        key.keyframe_insert(data_path="value", frame=FrameNumber)
                        break    
        # Set the value for a specific shape key. Replace 'ShapeKeyName' with the name of your shape key.
        # And set 'value' to a value between 0 (not influenced) and 1 (fully influenced).
        #shape_keys['browDown_L'].value = 0.5

        # Insert a keyframe for the shape key at a specific frame. Replace 'FrameNumber' with your desired frame.
        #shape_keys['browDown_L'].keyframe_insert(data_path="value", frame=FrameNumber)

    # Else, print an error if no shape keys are found
    else:
        print("Object does not have shape keys!")
