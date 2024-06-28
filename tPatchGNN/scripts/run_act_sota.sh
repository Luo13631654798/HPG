### tPatchGNN ###
patience=10
gpu=1


# 3000h-1000p
for seed in {1..5}
do
    python run_tpatchgnn.py \
    --dataset activity --state 'def' --history 3000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu
done

# 1000h-3000p
for seed in {1..5}
do
    python run_tpatchgnn.py \
    --dataset activity --state 'def' --history 1000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu
done


# 2000h-2000p
for seed in {1..5}
do
    python run_tpatchgnn.py \
    --dataset activity --state 'def' --history 2000 \
    --patience $patience --batch_size 32 --lr 1e-3 \
    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 32 \
    --outlayer Linear --seed $seed --gpu $gpu
done
