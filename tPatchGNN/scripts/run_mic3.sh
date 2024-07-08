### tPatchGNN ###
patience=10
gpu=1

#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 24 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 1.5 --stride 1.5 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

for seed in {1..5}
do
    python run_models.py \
    --dataset mimic --state def --history 24 \
    --patience 10 --batch_size 16 --lr 1e-3 \
    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 2 \
    --hid_dim 8 \
    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done



#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 36 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 6 --stride 6 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 6 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 12 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 6 --stride 6 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 6 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 36 \
#    --patience 10 --batch_size 8 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 12 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 36 \
#    --patience 10 --batch_size 8 --lr 1e-3 \
#    --patch_size 4.5 --stride 4.5 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 12 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 1.5 --stride 1.5 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 8 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 24 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 2 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 36 \
#    --patience 10 --batch_size 8 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 2 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset mimic --state def --history 12 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 2 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done