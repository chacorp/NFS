# python train.py \
#     --log_dir "ckpts" \
#     --max_epoch 500 \
#     --stage1 --tb \
#     --dec_type 'disp' \
#     --design 'new2' \
#     --seg_dim 20 \
#     --batch_size 1 \
#     --window_size 8

python train.py \
    --log_dir "ckpts" \
    --max_epoch 500 \
    --stage1 --tb \
    --dec_type 'disp' \
    --design 'new2' \
    --seg_dim 20 \
    --batch_size 8