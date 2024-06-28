### tPatchGNN ###
patience=10
gpu=0
# nohup sh ./scripts/run_ushcn_alpha.sh > ushcn_16P_888_AddNorm_alpha_search.log 2>&1 &
for i in $(awk 'BEGIN{for (i=0.1; i<=2.0; i+=0.2) printf "%.1f ", i}')
# for seed in 1
do
#    python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 192 --lr 1e-3 \
#    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 1 \
#    --outlayer Linear --seed 1 --gpu $gpu --alpha $i

    python run_models.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 4 --stride 4 --nhead 1 --tf_layer 1 --nlayer 3 \
    --te_dim 8 --node_dim 8 --hid_dim 8 \
    --outlayer Linear --seed 1 --gpu $gpu --alpha $i
done
