### tPatchGNN ###
patience=10
gpu=0
# nohup sh ./scripts/run_phy_alpha.sh > phy_1P_161616_AddNorm_alpha_search.log 2>&1 &

for i in $(awk 'BEGIN{for (i=0.1; i<=2.0; i+=0.2) printf "%.1f ", i}')
# for seed in 1
do
#    python run_models.py --dataset physionet --state 'def' --history 24 \
#    --patience $patience --batch_size 32 --lr 1e-3 \
#    --patch_size 8 --stride 8 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu
#    python run_models.py --dataset physionet --state 'def' --history 24 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 1 --outlayer Linear --seed 1 --gpu $gpu --alpha $i

#    python run_models.py --dataset physionet --state 'def' --history 24 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 1 --stride 1 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --te_dim 10 --node_dim 10 --hid_dim 1 --outlayer Linear --seed 1 --gpu $gpu --alpha $i

    python run_models.py --dataset physionet --state 'def' --history 24 \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 0.5 --stride 0.5 --nhead 1 --tf_layer 1 --nlayer 1 \
    --te_dim 16 --node_dim 16 --hid_dim 16 --outlayer Linear --seed 1 --gpu $gpu --alpha $i
done
