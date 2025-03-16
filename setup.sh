pip install -r requirements.txt

# # git clone https://github.com/MPI-IS/mesh.git
# ## change Makefile:L7 -> @pip install --no-deps --verbose --no-cache-dir .
# ## comment out requirements.txt -> # numpy pyopengl opencv-python
# cd mesh && make all && cd ..

## For the case when you have error installing pytorch3d ...
cp cpp_extension.py /usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
pip install easydict protobuf==3.20.0