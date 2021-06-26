#!/bin/bash

python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=0 --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=0.16 --weight_decay=0   --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir=gs://martin_ma_mql_simclr/chi_0715_800_ft --checkpoint=gs://martin_ma_mql_simclr/chi_0715_800  --use_tpu=True --tpu_name=simclr-32 --train_summary_steps=0 --width_multiplier 1 --resnet_depth 50 --sk_ratio 0.0625 --loss_type chi --alpha 0.0 --beta 0.0 --gamma 1.0

