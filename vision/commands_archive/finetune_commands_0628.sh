#!/bin/bash

python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=0 --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=0.064 --weight_decay=0   --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir=gs://martin_ma_mql_simclr/debug_full_chi_0628_ft --checkpoint=gs://martin_ma_mql_simclr/debug_full_chi_0628  --use_tpu=True --tpu_name=simclr-128 --train_summary_steps=0 --width_multiplier 2 --resnet_depth 152 --sk_ratio 0.0625 --loss_type chi --alpha 0.0 --beta 0.0 --gamma 1.0
#INFO:tensorflow:Saving dict for global step 28151: contrast_loss = 0.0, contrastive_top_1_accuracy = 1.0, contrastive_top_5_accuracy = 1.0, global_step = 28151, label_top_1_accuracy = 0.77958, label_top_5_accuracy = 0.93502, loss = 1.0067966, re

