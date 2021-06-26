# Self-supervised Representation Learning with Relative Predictive Coding

This is a codebase for the vision experiments.

## Enviroment setup

For CIFAR-10/CIFAR-100 experiments, our code can run on a *single* GPU. It does not support multi-GPUs, for reasons such as global BatchNorm and contrastive loss across cores.

Our models are also trained with TPUs. It is recommended to run distributed training with TPUs when using our code for pretraining on ImageNet.

We recommend using conda to avoid compatibility issue:
```
conda create -n rpc_vision python=3.6  
conda activate rpc_vision  
pip install -r requirements.txt  
conda install cudatoolkit cudnn
```

## Pretraining and Fine-Tuning

First create a checkpoint directory:

```
mkdir checkpoint
```

To pretrain and finetune the model on CIFAR-10 with a *single* GPU, try the following command:

```
bash gpu_pretrain_finetune_cifar.sh
```

For different hyper-parameter specification, please change the corresponding parameter in file `gpu_pretrain_finetune_cifar.sh`.
<!--
To pretrain the model on ImageNet with Cloud TPUs, first check out the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Once you have created virtual machine with Cloud TPUs, and pre-downloaded the ImageNet data for [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012), please set the following enviroment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

The following command can be used to pretrain a ResNet-50 on ImageNet (which reflects the default hyperparameters in our paper):

```
python run.py --train_mode=pretrain \
  --train_batch_size=4096 --train_epochs=100 --temperature=0.1 \
  --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

A batch size of 4096 requires at least 32 TPUs. 100 epochs takes around 6 hours with 32 TPU v3s. Note that learning rate of 0.3 with `learning_rate_scaling=linear` is equivalent to that of 0.075 with `learning_rate_scaling=sqrt` when the batch size is 4096. However, using sqrt scaling allows it to train better when smaller batch size is used.

## Finetuning the linear head (linear eval)

To fine-tune a linear head (with a single GPU), try the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.0 \
  --train_epochs=100 --train_batch_size=512 --warmup_epochs=0 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --checkpoint=/tmp/simclr_test --model_dir=/tmp/simclr_test_ft --use_tpu=False
```

You can check the results using tensorboard, such as

```
python -m tensorboard.main --logdir=/tmp/simclr_test
```

As a reference, the above runs on CIFAR-10 should give you around 91% accuracy, though it can be further optimized.

For fine-tuning a linear head on ImageNet using Cloud TPUs, first set the `CHKPT_DIR` to pretrained model dir and set a new `MODEL_DIR`, then use the following command:

```
python run.py --mode=train_then_eval --train_mode=finetune \
  --fine_tune_after_block=4 --zero_init_logits_layer=True \
  --variable_schema='(?!global_step|(?:.*/|^)Momentum|head)' \
  --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=1e-6 \
  --train_epochs=90 --train_batch_size=4096 --warmup_epochs=0 \
  --dataset=imagenet2012 --image_size=224 --eval_split=validation \
  --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --checkpoint=$CHKPT_DIR \
  --use_tpu=True --tpu_name=$TPU_NAME --train_summary_steps=0
```

As a reference, the above runs on ImageNet should give you around 64.5% accuracy.
-->

## Cite

[RPC paper](https://arxiv.org/abs/2103.11275):

```
@article{tsai2021self,
  title={Self-supervised representation learning with relative predictive coding},
  author={Tsai, Yao-Hung Hubert and Ma, Martin Q and Yang, Muqiao and Zhao, Han and Morency, Louis-Philippe and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:2103.11275},
  year={2021}
}
```

This code base is adapted from [SimCLR](https://github.com/google-research/simclr). The major change is in the file `objective.py`
