### tPatchGNN ###
patience=10
gpu=1
# nohup sh ./scripts/run_mic3_alpha.sh > mic_1P_848_AddNorm_alpha_search.log 2>&1 &
for i in $(awk 'BEGIN{for (i=0.1; i<2.0; i+=0.2) printf "%.1f ", i}')
# for seed in 1
do
#    python run_models.py \
#    --dataset activity --state 'def' --history 3000 \
#    --patience $patience --batch_size 32 --lr 1e-3 \
#    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu
#    python run_models.py \
#    --dataset mimic --state def --history 24 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 1 \
#    --outlayer Linear --seed 1 --gpu $gpu --alpha $i


    python run_models.py \
    --dataset mimic --state def --history 24 \
    --patience 10 --batch_size 16 --lr 1e-3 \
    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 8 --node_dim 4 --hid_dim 8 \
    --outlayer Linear --seed 1 --gpu $gpu --alpha $i
done
