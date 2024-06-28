### tPatchGNN ###
patience=10
gpu=0

for seed in {1..5}
# for seed in 1
do
    python run_models.py \
    --dataset ushcn --state 'def' --history 24 \
    --patience $patience --batch_size 192 --lr 1e-3 \
    --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu
done
