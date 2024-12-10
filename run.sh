python3 -m torch.distributed.launch --nproc_per_node=8   train.py  data/imagenet   --epochs 300 --img-size 224 --drop-path 0.2 --lr 1e-3  --weight-decay 0.05 --aa rand-m9-mstd0.5-inc1 --warmup-lr 1e-6 --warmup-epochs 5  --output save --min-lr 1e-6  --model pvg_s    -b 104 --experiment None

