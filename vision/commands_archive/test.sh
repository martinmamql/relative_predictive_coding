#!/bin/bash
BS=$2
ALPHA=$3
BETA=$4
GAMMA=$5
LOSS_TYPE=chi
TEMP=128
MODEL_DIR="bs-$BS-alpha-$ALPHA-beta-$BETA-gamma-$GAMMA-temp-$TEMP-loss-$LOSS_TYPE"
FT_DIR="bs-$BS-alpha-$ALPHA-beta-$BETA-gamma-$GAMMA-temp-$TEMP-loss-$LOSS_TYPE-ft"
CUDA_VISIBLE_DEVICES=$1 python run.py --train_mode=pretrain \
  --train_batch_size=$BS --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=$TEMP \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=checkpoint/$MODEL_DIR --use_tpu=False --loss_type=$LOSS_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA

CUDA_VISIBLE_DEVICES=$1 python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=checkpoint/$MODEL_DIR --model_dir=checkpoint/$FT_DIR --use_tpu=False
