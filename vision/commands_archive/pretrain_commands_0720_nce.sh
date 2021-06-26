#!/bin/bash
TPU_NAME=simclr-32
STORAGE_BUCKET=gs://martin_ma_mql_simclr
DATA_DIR=$STORAGE_BUCKET/tensorflow_datasets
MODEL_DIR=$STORAGE_BUCKET/nce_0720_100

python run.py --train_mode=pretrain \
	      --train_batch_size=4096 \
	      --train_epochs=100 \
	      --temperature=0.1 \
	      --learning_rate=0.1 \
	      --learning_rate_scaling=sqrt \
	      --weight_decay=1e-4 \
	      --dataset=imagenet2012 \
	      --image_size=224 \
	      --eval_split=validation \
	      --data_dir=$DATA_DIR \
	      --model_dir=$MODEL_DIR \
              --use_tpu=True \
	      --train_summary_steps=0 \
	      --sk_ratio 0.0625 \
	      --width_multiplier 1 \
	      --resnet_depth 50 \
	      --loss_type nce \
	      --tpu_name $TPU_NAME

