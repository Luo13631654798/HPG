### tPatchGNN ###
patience=10
gpu=0


for seed in {1..5}
do
    python run_models.py --dataset physionet --state def --history 36 \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 1 --tf_layer 1 --nlayer 1 \
    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python run_models.py --dataset physionet --state def --history 12 \
    --patience 10 --batch_size 64 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 1 --tf_layer 1 --nlayer 1 \
    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python run_models.py --dataset physionet --state def --history 36 \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 9 --stride 9 --nhead 1 --tf_layer 1 --nlayer 1 \
    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python run_models.py --dataset physionet --state def --history 12 \
    --patience 10 --batch_size 64 --lr 1e-3 \
    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 1 \
    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
done

#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 24 \
#    --patience 10 --batch_size 64 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 2 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done


#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 36 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 12 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 36 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 12 --stride 12 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 12 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 12 --stride 12 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 36 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 18 --stride 18 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 12 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 6 --stride 6 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 16 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 24 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 4 --stride 4 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {5..5}
#do
#    python run_models.py --dataset physionet --state def --history 36 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 4 --stride 4 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done
#
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 12 \
#    --patience 10 --batch_size 32 --lr 1e-3 \
#    --patch_size 4 --stride 4 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

# WOHie
#for seed in {1..5}
#do
#    python run_models.py --dataset physionet --state def --history 24 \
#    --patience 10 --batch_size 16 --lr 1e-3 \
#    --patch_size 24 --stride 24 --nhead 1 --tf_layer 1 --nlayer 1 \
#    --hid_dim 64 --outlayer Linear --seed $seed --gpu $gpu --alpha 1
#done

