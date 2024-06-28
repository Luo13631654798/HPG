### tPatchGNN ###
patience=10
gpu=0

for seed in {1..5}
# for seed in 1
do
    python run_models.py --dataset physionet --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu

    python run_models.py --dataset physionet --state 'def' --history 24 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed 1 --gpu 0

    python run_models.py \
    --dataset mimic --state 'def' --history 24 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu

    python run_models.py --dataset mimic --state 'def' --history 24 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 64 --outlayer Linear --seed 1 --gpu 0

    python run_models.py \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu 

    python run_models.py --dataset activity --state 'def' --history 3000 --patience 10 --batch_size 32 --lr 1e-3 --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 32 --outlayer Linear --seed 1 --gpu 0

    python run_models.py \
    --dataset ushcn --state 'def' --history 24 \
    --patience $patience --batch_size 192 --lr 1e-3 \
    --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu


    nohup python -u run_tPatchGNN.py --dataset ushcn --state 'def' --history 24 --patience 10 --batch_size 192 --lr 1e-3 --patch_size 2 --stride 2 --nhead 1 --tf_layer 1 --nlayer 1 --te_dim 10 --node_dim 10 --hid_dim 32 --outlayer Linear --seed 1 --gpu 0 > tPatchGNN_ushcn.log 2>&1 &
done
