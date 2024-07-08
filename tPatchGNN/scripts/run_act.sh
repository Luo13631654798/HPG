### tPatchGNN ###
patience=10
gpu=0

#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 3000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 3000 --stride 3000 --nhead 1 --tf_layer 1 --nlayer 2 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 3000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done


for seed in {1..5}
do
    python run_models.py \
    --dataset activity --state def --history 3000 \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 3 \
    --hid_dim 8 \
    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done


#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 2000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 1000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 2000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 1000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 200 --stride 200 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 2000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 134 --stride 134 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 1000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 67 --stride 67 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 32 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#Origin
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 3000 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 1000 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 2000 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 300 --stride 300 --nhead 1 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#WOHie
#for seed in {1..5}
#do
#    python run_models.py \
#    --dataset activity --state def --history 3000 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 3000 --stride 3000 --nhead 1 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
