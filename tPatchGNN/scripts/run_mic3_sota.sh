### tPatchGNN ###
patience=10
gpu=1

# 24h-24p
#for seed in {1..5}
#do
#    python run_tpatchgnn.py \
#    --dataset mimic --state 'def' --history 24 \
#    --patience $patience --batch_size 32 --lr 1e-3 \
#    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu
#done

# 36h-12p
for seed in {1..5}
do
    python run_tpatchgnn.py \
    --dataset mimic --state 'def' --history 36 \
    --patience $patience --batch_size 16 --lr 1e-3 \
    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 10 --node_dim 10 --hid_dim 64 \
    --outlayer Linear --seed $seed --gpu $gpu
done

# 12h-36p
#for seed in {1..5}
#do
#    python run_tpatchgnn.py \
#    --dataset mimic --state 'def' --history 12 \
#    --patience $patience --batch_size 32 --lr 1e-3 \
#    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu
#done
