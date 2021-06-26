#!/bin/bash
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>

python run.py --train_mode=pretrain \
	  --train_batch_size=4096 --train_epochs=100 --temperature=0.1 \
	    --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
	      --dataset=imagenet2012 --image_size=224 --eval_split=validation \
	        --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
		  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
