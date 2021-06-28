#!/bin/bash
# hyperparameters
STORAGE_BUCKET="checkpoint"
BATCH_SIZE=512
EPOCH=100
TEMP=128
LR=1.0
LR_SCALE='linear'
W_DECAY=1e-4
DATASET='cifar10'
IMAGE_SIZE=32
SK_RATIO=0.0
WIDTH_MUL=1
RESNET_DEPTH=18

# different for different losses
LOSS_TYPE='chi'
HIDDEN_NORM=True
if [ $LOSS_TYPE == 'chi' ]
then
# check hidden norm
HIDDEN_NORM=False
ALPHA=0.0
BETA=0.001
GAMMA=1.0
MODEL_DIR=$STORAGE_BUCKET/"${LOSS_TYPE}_BS_${BATCH_SIZE}_EPOCH_${EPOCH}_TEMP_${TEMP}_LR_${LR}_LRSCALE_${LR_SCALE}_WDECAY_${W_DECAY}_DATASET_${DATASET}_IMAGE_SIZE_${IMAGE_SIZE}_SKRATIO_${SK_RATIO}_WIDTHMUL_${WIDTH_MUL}_RESNETDEP_${RESNET_DEPTH}_HIDDENNORM_${HIDDEN_NORM}_ALPHA_${ALPHA}_BETA_${BETA}_GAMMA_${GAMMA}"
CUDA_VISIBLE_DEVICES=0 python run.py --train_mode=pretrain \
	      --train_batch_size=$BATCH_SIZE \
	      --train_epochs=$EPOCH \
	      --temperature=$TEMP \
	      --learning_rate=$LR \
	      --learning_rate_scaling=$LR_SCALE \
	      --weight_decay=$W_DECAY \
	      --dataset=$DATASET \
	      --image_size=$IMAGE_SIZE \
	      --eval_split=test \
	      --model_dir=$MODEL_DIR \
              --use_tpu=False \
	      --train_summary_steps=0 \
	      --sk_ratio $SK_RATIO \
	      --width_multiplier $WIDTH_MUL \
	      --resnet_depth $RESNET_DEPTH \
	      --loss_type $LOSS_TYPE \
	      --alpha=$ALPHA \
	      --beta=$BETA \
	      --gamma=$GAMMA \
	      --hidden_norm=$HIDDEN_NORM

# NCE, JS, WPC, etc
else
TEMP=0.5
MODEL_DIR=$STORAGE_BUCKET/"${LOSS_TYPE}_BS_${BATCH_SIZE}_EPOCH_${EPOCH}_TEMP_${TEMP}_LR_${LR}_LRSCALE_${LR_SCALE}_WDECAY_${W_DECAY}_DATASET_${DATASET}_IMAGE_SIZE_${IMAGE_SIZE}_SKRATIO_${SK_RATIO}_WIDTHMUL_${WIDTH_MUL}_RESNETDEP_${RESNET_DEPTH}_HIDDENNORM_${HIDDEN_NORM}"
CUDA_VISIBLE_DEVICES=0 python run.py --train_mode=pretrain \
	      --train_batch_size=$BATCH_SIZE \
	      --train_epochs=$EPOCH \
	      --temperature=$TEMP \
	      --learning_rate=$LR \
	      --learning_rate_scaling=$LR_SCALE \
	      --weight_decay=$W_DECAY \
	      --dataset=$DATASET \
	      --image_size=$IMAGE_SIZE \
	      --eval_split=test \
	      --model_dir=$MODEL_DIR \
              --use_tpu=False \
	      --train_summary_steps=0 \
	      --sk_ratio $SK_RATIO \
	      --width_multiplier $WIDTH_MUL \
	      --resnet_depth $RESNET_DEPTH \
	      --loss_type $LOSS_TYPE \
	      --hidden_norm=$HIDDEN_NORM

fi

##############################################################################################
#####################Fine tune
##############################################################################################
CHKPT_DIR=$MODEL_DIR
FINETUNE_AFTER_BLOCK=0
LR=0.1
WD=0
EPOCHS=100
WARMUP_EPOCHS=0
MODEL_DIR="${CHKPT_DIR}_ft_BS_${BATCH_SIZE}_FINETUNE_AFTER_BLOCK_${FINETUNE_AFTER_BLOCK}_LR_${LR}_WD_${WD}_EPOCH_${EPOCHS}_WARMUP_EPOCHS_${WARMUP_EPOCHS}"
echo $MODEL_DIR
if [ $LOSS_TYPE == "chi" ]
then
	echo "Running chi"
	CUDA_VISIBLE_DEVICES=0 python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=$DATASET --image_size=$IMAGE_SIZE --eval_split=test --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR  --use_tpu=False --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE --alpha $ALPHA --beta $BETA --gamma $GAMMA
else
	CUDA_VISIBLE_DEVICES=0 python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=$DATASET --image_size=$IMAGE_SIZE --eval_split=test --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR  --use_tpu=False --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE
fi
