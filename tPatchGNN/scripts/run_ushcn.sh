### tPatchGNN ###
patience=10
gpu=0

for seed in {1..5}
do


    python run_models.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 4 --tf_layer 1 --nlayer 3 \
    --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do


    python run_models.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 0.75 --stride 0.75 --nhead 4 --tf_layer 1 --nlayer 3 \
    --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done