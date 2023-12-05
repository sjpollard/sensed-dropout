torchrun --nproc_per_node=2 train.py --model sensed_dropout_vit_b_16 --train-sampling oracle\
    --inference-sampling oracle --epochs 100 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp\
    --label-smoothing 0.11 --clip-grad-norm 1 --device cuda --print-freq 100 --log-freq 10 --output-dir ''

torchrun --nproc_per_node=2 train.py --model sensed_dropout_vit_b_16 --tokens 32 --train-sampling random\
    --inference-sampling oracle --epochs 100 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp\
    --label-smoothing 0.11 --clip-grad-norm 1 --device cuda --print-freq 100 --log-freq 10 --output-dir ''

torchrun --nproc_per_node=2 train.py --model sensed_dropout_vit_b_16 --tokens 16 --train-sampling random\
    --inference-sampling oracle --epochs 100 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp\
    --label-smoothing 0.11 --clip-grad-norm 1 --device cuda --print-freq 100 --log-freq 10 --output-dir ''

torchrun --nproc_per_node=2 train.py --model sensed_dropout_vit_b_16 --tokens 8 --train-sampling random\
    --inference-sampling oracle --epochs 100 --batch-size 128 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 --lr-warmup-decay 0.033 --amp\
    --label-smoothing 0.11 --clip-grad-norm 1 --device cuda --print-freq 100 --log-freq 10 --output-dir ''