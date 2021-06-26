#!/bin/bash
TPU_NAME=simclr-128
STORAGE_BUCKET=gs://martin_ma_mql_simclr
DATA_DIR=$STORAGE_BUCKET/tensorflow_datasets
MODEL_DIR=$STORAGE_BUCKET/chi_0709_800

python run.py --train_mode=pretrain \
	      --train_batch_size=4096 \
	      --train_epochs=800 \
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
	      --tpu_name=$TPU_NAME \
	      --train_summary_steps=0 \
	      --sk_ratio 0.0625 \
	      --width_multiplier 2 \
	      --resnet_depth 152 \
	      --loss_type chi \
	      --alpha=0.0 \
	      --beta=0.0 \
	      --gamma=1.0

