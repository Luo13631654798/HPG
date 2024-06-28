### tPatchGNN ###
patience=10
gpu=1
#for (( h = 8; h <= 64; h=h*2 )); do
#  for (( p = 150; p <= 1200; p = p * 2 )); do
#    for (( l = 1; l <= 3; l++ )); do
##      python -u Main_warp.py --data_path ./path/to/datasets/ \
##--batch 32 --lr 1e-3 --epoch 50 --patience 5 --log ./path/to/log/ \
##--save_path ./path/to/save/ --task active --seed 0 --warp_num 0_1.2_1 \
##--history 227 --patch_size $p --stride $p --nlayer $l \
##--hid_dim $h --alpha 0.1 --gpu 1
#
#      python run_models.py \
#      --dataset activity --state def --history 3000 \
#      --patience 10 --batch_size 32 --lr 1e-3 \
#      --patch_size $p --stride $p --nhead 1 --tf_layer 1 --nlayer $l \
#      --hid_dim $h \
#      --outlayer Linear --seed 1 --gpu 1 --alpha 0.1
#    done
#
#  done
#
#done

#for (( p = 300; p <= 1200; p = p * 2 )); do
#  for (( h = 8; h <= 64; h=h*2 )); do
##    for (( l = 1; l <= 3; l++ )); do
##      python -u Main_warp.py --data_path ./path/to/datasets/ \
##--batch 32 --lr 1e-3 --epoch 50 --patience 5 --log ./path/to/log/ \
##--save_path ./path/to/save/ --task active --seed 0 --warp_num 0_1.2_1 \
##--history 227 --patch_size $p --stride $p --nlayer $l \
##--hid_dim $h --alpha 0.1 --gpu 1
#
#      python run_models.py \
#      --dataset activity --state def --history 3000 \
#      --patience 10 --batch_size 32 --lr 1e-3 \
#      --patch_size $p --stride $p --nhead 1 --tf_layer 1 --nlayer 3 \
#      --hid_dim $h \
#      --outlayer Linear --seed 1 --gpu 1 --alpha 0.1
#
#  done
#
#done

for i in $(awk 'BEGIN{for (i=0.1; i<=1; i+=0.2) printf "%.1f ", i}')
do
  for p in 150 200 300; do
#    for (( l = 1; l <= 3; l++ )); do
#      python -u Main_warp.py --data_path ./path/to/datasets/ \
#--batch 32 --lr 1e-3 --epoch 50 --patience 5 --log ./path/to/log/ \
#--save_path ./path/to/save/ --task active --seed 0 --warp_num 0_1.2_1 \
#--history 227 --patch_size $p --stride $p --nlayer $l \
#--hid_dim $h --alpha 0.1 --gpu 1

      python run_models.py \
    --dataset activity --state def --history 1000 \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size $p --stride $p --nhead 1 --tf_layer 1 --nlayer 3 \
    --hid_dim 64 \
    --outlayer Linear --seed 1 --gpu 1 --alpha $i

  done

done