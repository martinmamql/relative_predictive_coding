#!/bin/bash
TPU_NAME=simclr-128
STORAGE_BUCKET=gs://martin_ma_mql_simclr
DATA_DIR=$STORAGE_BUCKET/tensorflow_datasets

# hyperparameters
BATCH_SIZE=4096
EPOCH=100
TEMP=256
LR=0.1
LR_SCALE='sqrt'
W_DECAY=1e-4
DATASET='imagenet2012'
IMAGE_SIZE=224
SK_RATIO=0.0625
WIDTH_MUL=2
RESNET_DEPTH=152

# different for different losses
LOSS_TYPE='chi'
HIDDEN_NORM=false
if [ $LOSS_TYPE == 'chi' ]
then
# check hidden norm
HIDDEN_NORM=false
ALPHA=1.0
BETA=1e-4
GAMMA=1.0
if [ $TEMP -le 1 ]
then
	exit 0
fi
MODEL_DIR=$STORAGE_BUCKET/"${LOSS_TYPE}_BS_${BATCH_SIZE}_EPOCH_${EPOCH}_TEMP_${TEMP}_LR_${LR}_LRSCALE_${LR_SCALE}_WDECAY_${W_DECAY}_DATASET_${DATASET}_IMAGE_SIZE_${IMAGE_SIZE}_SKRATIO_${SK_RATIO}_WIDTHMUL_${WIDTH_MUL}_RESNETDEP_${RESNET_DEPTH}_HIDDENNORM_${HIDDEN_NORM}_ALPHA_${ALPHA}_BETA_${BETA}_GAMMA_${GAMMA}"
python run.py --train_mode=pretrain \
	      --train_batch_size=$BATCH_SIZE \
	      --train_epochs=$EPOCH \
	      --temperature=$TEMP \
	      --learning_rate=$LR \
	      --learning_rate_scaling=$LR_SCALE \
	      --weight_decay=$W_DECAY \
	      --dataset=$DATASET \
	      --image_size=$IMAGE_SIZE \
	      --eval_split=validation \
	      --data_dir=$DATA_DIR \
	      --model_dir=$MODEL_DIR \
              --use_tpu=True \
	      --tpu_name=$TPU_NAME \
	      --train_summary_steps=0 \
	      --sk_ratio $SK_RATIO \
	      --width_multiplier $WIDTH_MUL \
	      --resnet_depth $RESNET_DEPTH \
	      --loss_type $LOSS_TYPE \
	      --alpha=$ALPHA \
	      --beta=$BETA \
	      --gamma=$GAMMA \
	      --hidden_norm=$HIDDEN_NORM
fi

if [ $LOSS_TYPE == 'nce' ] # JS, WPC, etc
then
MODEL_DIR=$STORAGE_BUCKET/"${LOSS_TYPE}_BS_${BATCH_SIZE}_EPOCH_${EPOCH}_TEMP_${TEMP}_LR_${LR}_LRSCALE_${LR_SCALE}_WDECAY_${W_DECAY}_DATASET_${DATASET}_IMAGE_SIZE_${IMAGE_SIZE}_SKRATIO_${SK_RATIO}_WIDTHMUL_${WIDTH_MUL}_RESNETDEP_${RESNET_DEPTH}_HIDDENNORM_${HIDDEN_NORM}"
echo $MODEL_DIR
python run.py --train_mode=pretrain \
	      --train_batch_size=$BATCH_SIZE \
	      --train_epochs=$EPOCH \
	      --temperature=$TEMP \
	      --learning_rate=$LR \
	      --learning_rate_scaling=$LR_SCALE \
	      --weight_decay=$W_DECAY \
	      --dataset=$DATASET \
	      --image_size=$IMAGE_SIZE \
	      --eval_split=validation \
	      --data_dir=$DATA_DIR \
	      --model_dir=$MODEL_DIR \
              --use_tpu=True \
	      --tpu_name=$TPU_NAME \
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
LR=0.16
WD=0
EPOCHS=90
WARMUP_EPOCHS=0
MODEL_DIR="${CHKPT_DIR}_ft_BS_${BATCH_SIZE}_FINETUNE_AFTER_BLOCK_${FINETUNE_AFTER_BLOCK}_LR_${LR}_WD_${WD}_EPOCH_${EPOCHS}_WARMUP_EPOCHS_${WARMUP_EPOCHS}"
echo $MODEL_DIR
if [ $LOSS_TYPE == "chi" ]
then
	echo "Running chi"
	python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE --alpha $ALPHA --beta $BETA --gamma $GAMMA
else
	python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=$FINETUNE_AFTER_BLOCK --zero_init_logits_layer=True   --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)'   --global_bn=True --optimizer=momentum --learning_rate=$LR --weight_decay=$WD   --train_epochs=$EPOCHS --train_batch_size=$BATCH_SIZE --warmup_epochs=$WARMUP_EPOCHS   --dataset=imagenet2012 --image_size=224 --eval_split=validation   --data_dir=gs://martin_ma_mql_simclr/tensorflow_datasets --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0 --width_multiplier $WIDTH_MUL --resnet_depth $RESNET_DEPTH --sk_ratio $SK_RATIO --loss_type $LOSS_TYPE
fi
