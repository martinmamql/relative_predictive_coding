#!/bin/bash
TPU_NAME=simclr-32
CHKPT_DIR=chi_BS_4096_EPOCH_100_TEMP_256_LR_0.1_LRSCALE_sqrt_WDECAY_1e-4_DATASET_imagenet2012_IMAGE_SIZE_224_SKRATIO_0.0625_WIDTHMUL_1_RESNETDEP_50_HIDDENNORM_false_ALPHA_0.0_BETA_0.0_GAMMA_1.0
LOSS_TYPE=chi
WIDTH_MUL=1
RESNET_DEPTH=50
if [ $LOSS_TYPE == "chi" ]
then
	ALPHA=0.0
	BETA=0.0
	GAMMA=1.0
fi

SK_RATIO=0.0625
BATCH_SIZE=4096
FINETUNE_AFTER_BLOCK=0
LR=0.16
WD=0
EPOCHS=90
WARMUP_EPOCHS=0
MODEL_DIR="${CHKPT_DIR}_ft_BS_${BATCH_SIZE}_FINETUNE_AFTER_BLOCK_${FINETUNE_AFTER_BLOCK}_LR_${LR}_WD_${WD}_EPOCH_${EPOCHS}_WARMUP_EPOCHS_${WARMUP_EPOCHS}"
echo $MODEL_DIR
if [ $LOSS_TYPE == "chi" ]
then
	echo "Running chi"
	python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir="gs://martin_ma_mql_simclr/$MODEL_DIR" --checkpoint="gs://martin_ma_mql_simclr/$CHKPT_DIR"  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE --alpha $ALPHA --beta $BETA --gamma $GAMMA
else
	python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir="gs://martin_ma_mql_simclr/$MODEL_DIR" --checkpoint="gs://martin_ma_mql_simclr/$CHKPT_DIR"  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE
fi
