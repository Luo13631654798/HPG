patience=10
gpu=3

for seed in $(seq 1 5)
do
    for history in 24 36 12
    do
        python run_baselines.py  \
            --patience $patience --gpu $gpu --dataset "physionet" \
            --history $history --model "MSGNet"
    done
done

for seed in $(seq 1 5)
do
    for history in 24 36 12
    do
        python run_baselines.py  \
            --patience $patience --gpu $gpu --dataset "mimic" \
            --history $history --model "MSGNet"
    done
done


for seed in $(seq 1 5)
do
    for history in 3000 2000 1000
    do
        python run_baselines.py  \
            --patience $patience --gpu $gpu --dataset "activity" \
            --history $history --model "MSGNet"
    done
done

python run_baselines.py  \
        --patience $patience --gpu $gpu --dataset "ushcn" \
        --history 24 --model "MSGNet"

        