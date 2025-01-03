# Neural Face Skinning for Mesh-agnostic Facial Expression Cloning

<img src="assets/teaser.png" alt="drawing"/> 
<!-- <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" height=22.5></a> -->
<a href="https://chacorp.github.io/nfs-page/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=22.5></a>

This is official implementation of the paper 'Neural Face Skinning for Mesh-agnostic Facial Expression Cloning'

## TODOs
- [x] Installation
- [x] Inference
- [x] Train code
- [ ] Data preparation
- [ ] Dataloader for custom data
- [ ] Evaluation
- [ ] Pretrained model



## 1. Installation
### Environment
- System requirement: Ubuntu 20.04/Ubuntu 22.04, Cuda 11.8
- Tested GPU: NVIDIA-RTX-A5000

Use docker image `chacorp/audio2face:1.0`

```bash
docker pull chacorp/audio2face:1.0
```

All dependencies for the environment are provided in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Downloads
Download the following files from the link and place them in the root directory of this repo.
- [ICT files](https://drive.google.com/file/d/1NeSJyVgybzZS-p6uHafv6e3Tv8jTxbCy/view?usp=sharing). \
Place them in the root directory of this repo.
- [NFR files](https://drive.google.com/file/d/1cXXeU3AtpoGEVz2mhlWTSG1dEbAtCmD1/view?usp=sharing). \
Place them in the root directory of this repo.

```text
NFS/
  ├─ ...
  ├─ ict_face_pt/     # ICT files
  │  ├─ exp_basis.pt
  │  ├─ id_basis.pt
  │  ├─ neutral_verts.pt
  │  └─ quad_faces.pt
  ├─ data/           # NFR files
  ├─ experiments/    # NFR files
  ├─ test-mesh/      # NFR files
  └─ third_party/    # NFR files
```

### Pretrained model
TBD ...


## 2. Data preparation
### Processing data
To train model, data needs to be pre-processed. \
First, align your mesh to the examples in blender via the `align.blend` file provided from [NFR](https://github.com/dafei-qin/NFR_pytorch).

After the mesh alignment, use `utils/data_prepare.py` to obtain the gradient operator and precomputes for DiffusionNet (mass, Laplacian, spatial gradients, eigenbases ...) for the mesh.

※Note※ \
you only need to process the neutral face of the mesh! no need to process all expression meshes.

```bash
python utils/data_prepare.py \
  -i ${path_to_your_neutral_mesh} \
  -o ${path_to_processed_data}
```

The input and output directory would look like this:
```bash
${path_to_your_neutral_mesh}/
  └─ m00.obj            # aligned mesh using `align.blend`

${path_to_processed_data}/
  ├─ m00_dfn_info.pkl   # precomputes for DiffusionNet
  ├─ m00_img.npy        # rendered image (RGBD)
  ├─ m00_mesh.obj       # processed mesh
  └─ m00_operators.pkl  # gradient operator
```

### Dataloader
TBD ...


## 3. Inference
We provide a jupyter notebook for the inference. `notebook/NFS_inference.ipynb`. \
We also provide a jupyter notebook for NFR inference. `notebook/NFR_inference.ipynb`

## 4. Training
After the data preparation, you can train your own model by using bash file `train.sh` as below:
```bash
bash train.sh
```

The bash file can be modified for desired cases.
```shell
# train model from the scratch
python train.py \
  --log_dir "ckpts" \
  --max_epoch 500 \
  --stage1 --tb \
  --dec_type 'disp' \ 
  --design 'new2' \
  --seg_dim 20 \
  --warmup \
  --window_size 16

# continue or fine-tune model from the saved checkpoint
python train.py \
  --log_dir "ckpts" \
  --max_epoch 500 \
  --stage1 --tb \
  --dec_type 'disp' \
  --design 'new2' \
  --seg_dim 20 \
  --window_size 16 \
  --ckpt "ckpts/2025-01-02-06-54-46-all" # path to saved checkpoint
```


## 5. Evaluation
TBD ... 


## Acknowledgement

We would like to thank the contributors to the [NFR](https://github.com/dafei-qin/NFR_pytorch), [CodeTalker](https://github.com/Doubiiu/CodeTalker), [FaceFormer](https://github.com/EvelynFan/FaceFormer) and [Diffusion-Net](https://github.com/nmwsharp/diffusion-net) for their open research.

We also thank the contributors to the [ICT](https://github.com/ICT-VGL/ICT-FaceKit), [Multiface](https://github.com/facebookresearch/multiface), [VOCA](https://voca.is.tue.mpg.de/) and [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) for their open data.

<!-- ## Citation
TBD ... -->