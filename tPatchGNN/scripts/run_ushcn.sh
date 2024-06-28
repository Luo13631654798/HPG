### tPatchGNN ###
patience=10
gpu=0

for seed in {2..5}
# for seed in 1
do
#    python run_models.py \
#    --dataset ushcn --state 'def' --history 24 \
#    --patience $patience --batch_size 192 --lr 1e-3 \
#    --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu

#    python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 192 --lr 1e-3 \
#    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1

#    python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 128 --lr 1e-3 \
#    --patch_size 4 --stride 4 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 0.1

#    python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 128 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 4 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 0.3

    python run_models.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 3 --stride 3 --nhead 4 --tf_layer 1 --nlayer 3 \
    --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu --alpha 0.95
done
