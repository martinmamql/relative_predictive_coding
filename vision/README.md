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

To pretrain the model on ImageNet with Cloud TPUs, first check out the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist) for basic information on how to use Google Cloud TPUs.

Once you have created virtual machine with Cloud TPUs, and pre-downloaded the ImageNet data for [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/imagenet2012), please set the following enviroment variables:

```
TPU_NAME=<tpu-name>
STORAGE_BUCKET=gs://<storage-bucket>
DATA_DIR=$STORAGE_BUCKET/<path-to-tensorflow-dataset>
MODEL_DIR=$STORAGE_BUCKET/<path-to-store-checkpoints>
```

in the following files which pretrain and fine-tune a ResNet-50 or a ResNet-152 on ImageNet:

```
bash tpu_pretrain_finetune_resnet50.sh  
bash tpu_pretrain_finetune_resnet128.sh  
```

To request checkpoints of the trained models from the commands above, please contact Martin via qianlim@andrew.cmu.edu.  

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
