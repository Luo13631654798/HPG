### tPatchGNN ###
patience=10
gpu=1
#for (( p = 1; p <= 8; p = p * 2 )); do
#  for (( l = 1; l <= 3; l=l+1 )); do
#    for (( d = 16; d <= 64; d=d*2 )); do
#        python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 128 --lr 1e-3 \
#    --patch_size $p --stride $p --nhead 4 --tf_layer 1 --nlayer $l \
#    --hid_dim $d \
#    --outlayer Linear --seed 1 --gpu 1 --alpha 0.7
#
#  done
#done
#done

#for (( p = 3; p <= 4; p=p+1 ));
#do
#   for i in $(awk 'BEGIN{for (i=0.1; i<=1; i+=0.2) printf "%.1f ", i}'); do
##      python run_models.py \
##    --dataset ushcn --state def --history 24 \
##    --patience $patience --batch_size 128 --lr 1e-3 \
##    --patch_size 4 --stride 4 --nhead $h --tf_layer 1 --nlayer 3 \
##    --hid_dim 64 \
##    --outlayer Linear --seed 1 --gpu 1 --alpha $i
#
#      python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 128 --lr 1e-3 \
#    --patch_size $p --stride $p --nhead 4 --tf_layer 1 --nlayer 3 \
#    --hid_dim 64 \
#    --outlayer Linear --seed 1 --gpu 0 --alpha $i
#  done
#done

#for l in 4;
#do
#   for d in 16 32 64 128; do
##      python run_models.py \
##    --dataset ushcn --state def --history 24 \
##    --patience $patience --batch_size 128 --lr 1e-3 \
##    --patch_size 4 --stride 4 --nhead $h --tf_layer 1 --nlayer 3 \
##    --hid_dim 64 \
##    --outlayer Linear --seed 1 --gpu 1 --alpha $i
#
#      python run_models.py \
#    --dataset ushcn --state def --history 24 \
#    --patience $patience --batch_size 128 --lr 1e-3 \
#    --patch_size 3 --stride 3 --nhead 4 --tf_layer 1 --nlayer $l \
#    --hid_dim $d \
#    --outlayer Linear --seed 1 --gpu 0 --alpha 0.3
#  done
#done

for i in $(awk 'BEGIN{for (i=0.2; i<=1; i+=0.2) printf "%.1f ", i}'); do
  python run_models.py \
  --dataset ushcn --state def --history 24 \
  --patience $patience --batch_size 128 --lr 1e-3 \
  --patch_size 3 --stride 3 --nhead 4 --tf_layer 1 --nlayer 3 \
  --hid_dim 64 \
  --outlayer Linear --seed 1 --gpu 0 --alpha $i
done


#for b in 64 72 96 128 144 160 192; do
#  python run_models.py \
#--dataset ushcn --state def --history 24 \
#--patience $patience --batch_size $b --lr 1e-3 \
#--patch_size 3 --stride 3 --nhead 4 --tf_layer 1 --nlayer 3 \
#--hid_dim 64 \
#--outlayer Linear --seed 1 --gpu 0 --alpha 0.3
#done