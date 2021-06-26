python run.py --mode=train_then_eval --train_mode=finetune \
	  --fine_tune_after_block=4 --zero_init_logits_layer=True \
	    --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
	      --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=1e-6 \
	        --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0 \
		  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
		    --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
		      --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
